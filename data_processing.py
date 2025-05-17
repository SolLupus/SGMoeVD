# data_processing.py
import argparse
import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import re
import numpy as np
import random
import torch.nn.functional as F
from collections import Counter
from utils import get_codebert_embeddings, load_codebert_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VulnerabilityDataset(Dataset):
    """漏洞检测数据集类"""
    
    def __init__(self, file_path, max_length=512, node_feature_dim=128, is_valid=False, is_test=False, use_codebert=True):
        """
        初始化数据集
        
        参数:
            file_path: CSV文件路径，包含代码和标签
            max_length: 代码的最大token长度
            node_feature_dim: 图节点的特征维度
            is_valid: 是否为验证集
            is_test: 是否为测试集
            use_codebert: 是否使用CodeBERT进行向量化
        """
        self.file_path = file_path
        self.max_length = max_length
        self.node_feature_dim = node_feature_dim
        self.is_valid = is_valid
        self.is_test = is_test
        self.use_codebert = use_codebert
        
        if file_path == None:
            self.data  = None
        else:
            self.data = pd.read_csv(file_path)
        
        # 设置CodeBERT模型
        if use_codebert:
            # 使用GPU加载CodeBERT模型（如可用）
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.tokenizer, self.model = load_codebert_model('microsoft/codebert-base', device)
            self.embedding_dim = 768  # CodeBERT的嵌入维度
            print(f"CodeBERT模型已加载到 {device} 设备上")
        if self.data is not None:
            print(f"加载了 {len(self.data)} 个样本从 {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        code = self.data.iloc[idx]['code']
        label = self.data.iloc[idx]['label']
        teacher_label = self.data.iloc[idx]['teacher_label'] if 'teacher_label' in self.data.columns else label

        # 生成三种图: PDG, CPG, AST
        is_negative = label == 0
        pdg_graph = self._generate_pdg(code, is_negative)
        cpg_graph = self._generate_cpg(code, is_negative)
        ast_graph = self._generate_ast(code, is_negative)
        
        # 确保所有图数据都在CPU上，而不是CUDA上
        # PyG的DataLoader会通过pin_memory将数据从CPU迁移到GPU
        # 因此在这里需要确保数据最初是在CPU上的
        for graph in [pdg_graph, cpg_graph, ast_graph]:
            for key, value in graph:
                if torch.is_tensor(value) and value.is_cuda:
                    graph[key] = value.cpu()
        
        graphs = {
            'pdg': pdg_graph,
            'cpg': cpg_graph,
            'ast': ast_graph
        }

        return {
            'graphs': graphs,
            'labels': torch.tensor(label, dtype=torch.long),
            'teacher_labels': torch.tensor(teacher_label, dtype=torch.long)
        }
    
    def _generate_pdg(self, code, is_negative):
        """
        生成程序依赖图 (Program Dependency Graph)
        PDG表示程序中数据流和控制流依赖关系
        
        参数:
            code: 代码字符串
            is_negative: 是否为负样本（非漏洞代码）
        
        返回:
            PyG格式的图对象
        """
        # 1. 分词和预处理
        tokens = code.split()
        lines = code.split('\n')
        
        # 确定节点数量 (每个token作为一个节点)
        num_nodes = min(150, len(tokens))  # 限制最大节点数
        
        # 2. 使用CodeBERT获取节点特征
        if self.use_codebert:
            # 获取代码的CodeBERT嵌入
            token_embeddings = get_codebert_embeddings(
                code, 
                self.tokenizer, 
                self.model, 
                # max_length=self.max_length
            )
            
            # 选取前num_nodes个token的嵌入
            if token_embeddings.size(0) >= num_nodes:
                node_features = token_embeddings[:num_nodes]
            else:
                # 如果token数量不足，则填充
                padding = torch.zeros(num_nodes - token_embeddings.size(0), self.embedding_dim)
                node_features = torch.cat([token_embeddings, padding], dim=0)
        else:
            # 如果不使用CodeBERT，则生成手工特征
            node_features = torch.zeros(num_nodes, self.node_feature_dim)
            
            # 为PDG添加特有的安全漏洞相关特征
            for i in range(min(num_nodes, len(tokens))):
                token = tokens[i] if i < len(tokens) else ""
                
                # 基本特征
                is_keyword = 1.0 if token in ["if", "for", "while", "return", "switch", "case", "break"] else 0.0
                is_operator = 1.0 if token in ["+", "-", "*", "/", "=", "==", "!=", ">", "<", ">=", "<="] else 0.0
                is_bracket = 1.0 if token in ["{", "}", "(", ")", "[", "]"] else 0.0
                is_variable = 1.0 if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) and not is_keyword else 0.0
                is_function = 1.0 if i > 0 and tokens[i-1] in ["void", "int", "char", "float", "double"] else 0.0
                token_len = min(len(token), 20) / 20.0  # 归一化长度
                
                # PDG特有的漏洞检测特征
                is_pointer = 1.0 if "*" in token or token in ["malloc", "free", "calloc", "realloc"] else 0.0
                is_array_access = 1.0 if i < len(tokens) - 1 and tokens[i+1] == "[" else 0.0
                is_null_check = 1.0 if token in ["NULL", "null", "nullptr"] else 0.0
                is_memory_op = 1.0 if token in ["memcpy", "strcpy", "strcat", "malloc", "free"] else 0.0
                is_size_check = 1.0 if token in ["sizeof", "strlen", "size"] else 0.0
                
                # 设置特征向量
                feat_idx = 0
                node_features[i, feat_idx] = token_len; feat_idx += 1
                node_features[i, feat_idx] = is_keyword; feat_idx += 1
                node_features[i, feat_idx] = is_operator; feat_idx += 1
                node_features[i, feat_idx] = is_bracket; feat_idx += 1
                node_features[i, feat_idx] = is_variable; feat_idx += 1
                node_features[i, feat_idx] = is_function; feat_idx += 1
                node_features[i, feat_idx] = is_pointer; feat_idx += 1
                node_features[i, feat_idx] = is_array_access; feat_idx += 1
                node_features[i, feat_idx] = is_null_check; feat_idx += 1
                node_features[i, feat_idx] = is_memory_op; feat_idx += 1
                node_features[i, feat_idx] = is_size_check; feat_idx += 1
                
                # 添加位置信息
                node_features[i, feat_idx] = i / num_nodes; feat_idx += 1
                
                # 其余维度用随机值填充
                node_features[i, feat_idx:] = torch.randn(self.node_feature_dim - feat_idx)
        
        # 3. 生成边列表 - 数据依赖关系
        src_nodes = []
        dst_nodes = []
        edge_types = []  # 边的类型
        
        # 变量定义和使用之间的依赖
        variables = {}  # 记录变量最后一次出现的位置
        
        for i in range(min(num_nodes, len(tokens))):
            token = tokens[i]
            
            # 自连接
            src_nodes.append(i)
            dst_nodes.append(i)
            edge_types.append(0)  # 类型0: 自连接
            
            # 如果是变量名
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) and token not in ["if", "for", "while", "return", "switch", "case", "break"]:
                # 如果变量已经出现过，建立依赖关系
                if token in variables:
                    prev_pos = variables[token]
                    # 当前使用依赖于之前的定义
                    src_nodes.append(i)
                    dst_nodes.append(prev_pos)
                    edge_types.append(1)  # 类型1: 变量依赖
                    
                    # 双向连接
                    src_nodes.append(prev_pos)
                    dst_nodes.append(i)
                    edge_types.append(1)
                
                # 更新变量位置
                variables[token] = i
            
            # 赋值操作的依赖
            if token == "=" and i > 0 and i < len(tokens) - 1:
                left_var = tokens[i-1]
                right_var = tokens[i+1]
                
                # 右侧变量到左侧变量的依赖
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', right_var) and right_var in variables:
                    right_pos = variables[right_var]
                    # 左侧变量依赖于右侧变量
                    src_nodes.append(i-1)
                    dst_nodes.append(right_pos)
                    edge_types.append(2)  # 类型2: 赋值依赖
                    
                    # 双向连接
                    src_nodes.append(right_pos)
                    dst_nodes.append(i-1)
                    edge_types.append(2)
            
            # PDG特有的边：内存相关操作的依赖
            if token in ["malloc", "calloc", "realloc"] and i < len(tokens) - 3:
                # 寻找内存操作后的变量
                for j in range(i+1, min(i+5, len(tokens))):
                    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tokens[j]):
                        # 内存分配和变量之间的连接
                        src_nodes.append(i)
                        dst_nodes.append(j)
                        edge_types.append(3)  # 类型3: 内存分配依赖
                        break
            
            # 指针解引用与指针变量的连接
            if token == "*" and i > 0:
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tokens[i-1]) and tokens[i-1] in variables:
                    ptr_pos = variables[tokens[i-1]]
                    # 指针解引用依赖于指针变量
                    src_nodes.append(i)
                    dst_nodes.append(ptr_pos)
                    edge_types.append(4)  # 类型4: 指针解引用
        
        # 添加一些随机连接，以增强图连通性
        if num_nodes > 2:
            for i in range(num_nodes):
                # 统计节点的边数
                edge_count = 0
                for s, d in zip(src_nodes, dst_nodes):
                    if s == i or d == i:
                        edge_count += 1
                
                # 如果节点的边数不足2，添加随机连接
                if edge_count < 2:
                    j = (i + random.randint(1, num_nodes-1)) % num_nodes
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_types.append(5)  # 类型5: 随机连接
                    
                    src_nodes.append(j)
                    dst_nodes.append(i)
                    edge_types.append(5)
        
        # 4. 创建PyG图对象
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph
    
    def _generate_cpg(self, code, is_negative):
        """
        生成控制流图 (Control Flow Graph)
        CPG表示程序的控制流结构
        
        参数:
            code: 代码字符串
            is_negative: 是否为负样本
        
        返回:
            PyG格式的图对象
        """
        # 1. 分词和预处理
        tokens = code.split()
        lines = code.split('\n')
        
        # 确定节点数量 (每个token作为一个节点)
        num_nodes = min(150, len(tokens))  # 限制最大节点数
        
        # 2. 使用CodeBERT获取节点特征
        if self.use_codebert:
            # 获取代码的CodeBERT嵌入
            token_embeddings = get_codebert_embeddings(
                code, 
                self.tokenizer, 
                self.model, 
                # max_length=self.max_length
            )
            
            # 选取前num_nodes个token的嵌入
            if token_embeddings.size(0) >= num_nodes:
                node_features = token_embeddings[:num_nodes]
            else:
                # 如果token数量不足，则填充
                padding = torch.zeros(num_nodes - token_embeddings.size(0), self.embedding_dim)
                node_features = torch.cat([token_embeddings, padding], dim=0)
        else:
            # 如果不使用CodeBERT，则生成手工特征
            node_features = torch.zeros(num_nodes, self.node_feature_dim)
            
            # 为CFG添加特有的控制流和漏洞相关特征
            for i in range(min(num_nodes, len(tokens))):
                token = tokens[i] if i < len(tokens) else ""
                
                # 基于token的特征
                is_control = 1.0 if token in ["if", "for", "while", "switch", "case", "break", "continue", "return"] else 0.0
                is_bracket = 1.0 if token in ["{", "}", "(", ")", "[", "]"] else 0.0
                is_condition = 1.0 if token in ["==", "!=", ">", "<", ">=", "<=", "&&", "||", "!"] else 0.0
                token_len = min(len(token), 20) / 20.0  # 归一化长度
                
                # CFG特有的漏洞检测特征
                is_error_check = 1.0 if token in ["if", "assert"] and i < len(tokens) - 3 and any(err_token in tokens[i:i+4] for err_token in ["error", "fail", "invalid"]) else 0.0
                is_null_check = 1.0 if token in ["if", "assert"] and i < len(tokens) - 3 and any(null_token in tokens[i:i+4] for null_token in ["null", "NULL", "nullptr", "None"]) else 0.0
                is_bounds_check = 1.0 if token in ["if", "assert"] and i < len(tokens) - 3 and any(bound_op in tokens[i:i+4] for bound_op in ["<", ">", "<=", ">=", "==", "!="]) else 0.0
                is_early_return = 1.0 if token == "return" and i > 2 and tokens[i-2] == "if" else 0.0
                is_loop_control = 1.0 if token in ["for", "while"] and i < len(tokens) - 5 else 0.0
                
                # 设置特征向量
                feat_idx = 0
                node_features[i, feat_idx] = token_len; feat_idx += 1
                node_features[i, feat_idx] = i / num_nodes; feat_idx += 1  # 归一化位置
                node_features[i, feat_idx] = is_control; feat_idx += 1
                node_features[i, feat_idx] = is_bracket; feat_idx += 1
                node_features[i, feat_idx] = is_condition; feat_idx += 1
                node_features[i, feat_idx] = is_error_check; feat_idx += 1
                node_features[i, feat_idx] = is_null_check; feat_idx += 1
                node_features[i, feat_idx] = is_bounds_check; feat_idx += 1
                node_features[i, feat_idx] = is_early_return; feat_idx += 1
                node_features[i, feat_idx] = is_loop_control; feat_idx += 1
                
                # 其余维度用随机值填充
                node_features[i, feat_idx:] = torch.randn(self.node_feature_dim - feat_idx)
        
        # 3. 生成边列表 - 控制流关系
        src_nodes = []
        dst_nodes = []
        edge_types = []  # 边的类型
        
        # 连接相邻的token (顺序关系)
        for i in range(num_nodes - 1):
            src_nodes.append(i)
            dst_nodes.append(i+1)
            edge_types.append(0)  # 类型0: 顺序流
            
            src_nodes.append(i+1)
            dst_nodes.append(i)
            edge_types.append(0)  # 类型0: 顺序流
        
        # 自连接
        for i in range(num_nodes):
            src_nodes.append(i)
            dst_nodes.append(i)
            edge_types.append(1)  # 类型1: 自连接
        
        # 控制流结构
        control_stack = []  # 用于跟踪控制流结构
        
        for i in range(min(num_nodes, len(tokens))):
            token = tokens[i]
            
            # 控制流开始
            if token in ["if", "for", "while", "switch"]:
                control_stack.append((token, i))
            
            # 控制流结束 (假设右大括号标志着控制流结束)
            elif token == "}" and control_stack:
                control_type, start_idx = control_stack.pop()
                
                # 连接控制流的开始和结束
                src_nodes.append(start_idx)
                dst_nodes.append(i)
                edge_types.append(2)  # 类型2: 控制流连接
                
                src_nodes.append(i)
                dst_nodes.append(start_idx)
                edge_types.append(2)
                
                # 对于循环，添加从结束到开始的回边
                if control_type in ["for", "while"]:
                    src_nodes.append(i)
                    dst_nodes.append(start_idx)
                    edge_types.append(3)  # 类型3: 循环回边
            
            # 处理break和continue
            elif token == "break" and control_stack:
                # 找到最近的循环或switch
                for j in range(len(control_stack)-1, -1, -1):
                    if control_stack[j][0] in ["for", "while", "switch"]:
                        _, start_idx = control_stack[j]
                        src_nodes.append(i)
                        dst_nodes.append(start_idx)
                        edge_types.append(4)  # 类型4: break/continue
                        break
            
            elif token == "continue" and control_stack:
                # 找到最近的循环
                for j in range(len(control_stack)-1, -1, -1):
                    if control_stack[j][0] in ["for", "while"]:
                        _, start_idx = control_stack[j]
                        src_nodes.append(i)
                        dst_nodes.append(start_idx)
                        edge_types.append(4)
                        break
        
        # CFG特有的边：错误检查和条件分支
        for i in range(min(num_nodes, len(tokens))):
            token = tokens[i]
            
            # 错误检查与后续代码的关系
            if token == "if" and i < len(tokens) - 3:
                if any(err_token in tokens[i:i+5] for err_token in ["error", "fail", "invalid", "null", "NULL"]):
                    # 寻找if块的结尾
                    brace_count = 0
                    if_end = i
                    for j in range(i+1, min(i+20, len(tokens))):
                        if tokens[j] == "{":
                            brace_count += 1
                        elif tokens[j] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                if_end = j
                                break
                    
                    # 连接错误检查和之后的代码
                    if if_end < len(tokens) - 1 and if_end < num_nodes:
                        src_nodes.append(i)
                        dst_nodes.append(if_end+1)
                        edge_types.append(5)  # 类型5: 错误处理流
            
            # 循环条件与循环体的关系
            if token in ["for", "while"] and i < len(tokens) - 3:
                # 寻找循环体开始位置
                for j in range(i+1, min(i+10, len(tokens))):
                    if tokens[j] == "{":
                        # 连接循环条件与循环体开始
                        src_nodes.append(i)
                        dst_nodes.append(j)
                        edge_types.append(6)  # 类型6: 循环控制
                        break
        
        # 4. 创建PyG图对象
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph
    
    def _generate_ast(self, code, is_negative):
        """
        生成抽象语法树 (Abstract Syntax Tree)
        AST表示代码的语法结构
        
        参数:
            code: 代码字符串
            is_negative: 是否为负样本
        
        返回:
            PyG格式的图对象
        """
        # 1. 分词和预处理
        tokens = code.split()
        lines = code.split('\n')
        
        # 确定节点数量 (每个token作为一个节点)
        num_nodes = min(150, len(tokens))  # 限制最大节点数
        
        # 2. 使用CodeBERT获取节点特征
        if self.use_codebert:
            # 获取代码的CodeBERT嵌入
            token_embeddings = get_codebert_embeddings(
                code, 
                self.tokenizer, 
                self.model, 
                # max_length=self.max_length
            )
            
            # 选取前num_nodes个token的嵌入
            if token_embeddings.size(0) >= num_nodes:
                node_features = token_embeddings[:num_nodes]
            else:
                # 如果token数量不足，则填充
                padding = torch.zeros(num_nodes - token_embeddings.size(0), self.embedding_dim)
                node_features = torch.cat([token_embeddings, padding], dim=0)
        else:
            # 如果不使用CodeBERT，则生成手工特征
            node_features = torch.zeros(num_nodes, self.node_feature_dim)
            
            # 为AST添加特有的语法结构和漏洞相关特征
            for i in range(min(num_nodes, len(tokens))):
                token = tokens[i] if i < len(tokens) else ""
                
                # 基于token的特征 - AST节点类型
                is_keyword = 1.0 if token in ["if", "for", "while", "return", "switch", "case", "break"] else 0.0
                is_operator = 1.0 if token in ["+", "-", "*", "/", "=", "==", "!=", ">", "<", ">=", "<="] else 0.0
                is_bracket = 1.0 if token in ["{", "}", "(", ")", "[", "]"] else 0.0
                is_literal = 1.0 if token.isdigit() or token in ["true", "false", "null"] or token.startswith('"') or token.startswith("'") else 0.0
                is_identifier = 1.0 if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) and not is_keyword else 0.0
                token_len = min(len(token), 20) / 20.0  # 归一化长度
                
                # AST特有的漏洞检测特征
                is_func_decl = 1.0 if i > 0 and tokens[i-1] in ["void", "int", "char", "float", "double"] else 0.0
                is_param = 1.0 if i > 1 and tokens[i-2] == "(" and tokens[i-1] in ["int", "char", "float", "double"] else 0.0
                is_string_manipulation = 1.0 if token in ["strcpy", "strcat", "strcmp", "strlen", "sprintf", "gets"] else 0.0
                is_array_decl = 1.0 if i < len(tokens) - 1 and tokens[i+1] == "[" else 0.0
                is_memory_alloc = 1.0 if token in ["malloc", "calloc", "realloc", "new"] else 0.0
                
                # 设置特征向量
                feat_idx = 0
                node_features[i, feat_idx] = token_len; feat_idx += 1
                node_features[i, feat_idx] = is_keyword; feat_idx += 1
                node_features[i, feat_idx] = is_operator; feat_idx += 1
                node_features[i, feat_idx] = is_bracket; feat_idx += 1
                node_features[i, feat_idx] = is_literal; feat_idx += 1
                node_features[i, feat_idx] = is_identifier; feat_idx += 1
                node_features[i, feat_idx] = is_func_decl; feat_idx += 1
                node_features[i, feat_idx] = is_param; feat_idx += 1
                node_features[i, feat_idx] = is_string_manipulation; feat_idx += 1
                node_features[i, feat_idx] = is_array_decl; feat_idx += 1
                node_features[i, feat_idx] = is_memory_alloc; feat_idx += 1
                node_features[i, feat_idx] = i / num_nodes; feat_idx += 1  # 归一化位置
                
                # 其余维度用随机值填充
                node_features[i, feat_idx:] = torch.randn(self.node_feature_dim - feat_idx)
        
        # 3. 生成边列表 - 语法结构关系
        src_nodes = []
        dst_nodes = []
        edge_types = []  # 边的类型
        
        # 自连接
        for i in range(num_nodes):
            src_nodes.append(i)
            dst_nodes.append(i)
            edge_types.append(0)  # 类型0: 自连接
        
        # 括号匹配 (模拟AST的父子关系)
        brackets = []
        for i in range(min(num_nodes, len(tokens))):
            token = tokens[i]
            if token in ["{", "(", "["]:
                brackets.append((token, i))
            elif token in ["}", ")", "]"]:
                if brackets:
                    opening, idx = brackets.pop()
                    if (opening == "{" and token == "}") or \
                       (opening == "(" and token == ")") or \
                       (opening == "[" and token == "]"):
                        # 连接匹配的括号
                        src_nodes.append(idx)
                        dst_nodes.append(i)
                        edge_types.append(1)  # 类型1: 括号匹配
                        
                        src_nodes.append(i)
                        dst_nodes.append(idx)
                        edge_types.append(1)
                        
                        # 连接括号内的所有节点到开括号 (模拟AST的父子关系)
                        for j in range(idx + 1, i):
                            src_nodes.append(idx)
                            dst_nodes.append(j)
                            edge_types.append(2)  # 类型2: 父子关系
        
        # 关键字和其后的括号连接 (模拟AST的父子关系)
        for i in range(min(num_nodes - 1, len(tokens) - 1)):
            if tokens[i] in ["if", "for", "while", "switch"]:
                # 寻找最近的左括号
                for j in range(i + 1, min(i + 10, num_nodes)):
                    if j < len(tokens) and tokens[j] in ["(", "{"]:
                        src_nodes.append(i)
                        dst_nodes.append(j)
                        edge_types.append(3)  # 类型3: 关键字-括号关系
                        
                        src_nodes.append(j)
                        dst_nodes.append(i)
                        edge_types.append(3)
                        break
        
        # 函数声明和参数的关系
        for i in range(min(num_nodes - 2, len(tokens) - 2)):
            if tokens[i] in ["int", "void", "char", "float", "double"] and \
               re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tokens[i+1]) and \
               tokens[i+2] == "(":
                # 函数名和返回类型
                src_nodes.append(i)
                dst_nodes.append(i+1)
                edge_types.append(4)  # 类型4: 函数声明关系
                
                src_nodes.append(i+1)
                dst_nodes.append(i)
                edge_types.append(4)
                
                # 函数名和参数列表
                src_nodes.append(i+1)
                dst_nodes.append(i+2)
                edge_types.append(5)  # 类型5: 函数-参数关系
                
                src_nodes.append(i+2)
                dst_nodes.append(i+1)
                edge_types.append(5)
        
        # AST特有边：安全函数调用和参数的关系
        for i in range(min(num_nodes, len(tokens))):
            token = tokens[i]
            
            # 字符串操作和缓冲区安全性
            if token in ["strcpy", "strcat", "sprintf", "gets", "memcpy"]:
                # 搜索函数调用的参数
                if i < len(tokens) - 1 and tokens[i+1] == "(":
                    # 搜索第一个参数 (目标缓冲区)
                    for j in range(i+2, min(i+10, len(tokens))):
                        if j < len(tokens) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tokens[j]):
                            src_nodes.append(i)
                            dst_nodes.append(j)
                            edge_types.append(6)  # 类型6: 函数调用-参数关系
                            break
            
            # 内存分配和大小参数的关系
            if token in ["malloc", "calloc", "realloc"]:
                # 搜索大小参数
                if i < len(tokens) - 1 and tokens[i+1] == "(":
                    for j in range(i+2, min(i+10, len(tokens))):
                        if j < len(tokens) and (tokens[j].isdigit() or tokens[j] == "sizeof"):
                            src_nodes.append(i)
                            dst_nodes.append(j)
                            edge_types.append(7)  # 类型7: 内存分配-大小关系
                            break
        
        # 4. 创建PyG图对象
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
        
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph
    
    @staticmethod
    def collate_fn(batch):
        """将批次数据整理成模型所需的格式"""
        # 提取批次中的各个字段
        labels = torch.stack([item['labels'] for item in batch])
        teacher_labels = torch.stack([item['teacher_labels'] for item in batch])
        
        # 处理三种图数据
        batch_graphs = {
            'pdg': None,
            'cpg': None,
            'ast': None
        }
        
        # 对每种图分别处理
        for graph_type in ['pdg', 'cpg', 'ast']:
            # 使用PyG的Batch.from_data_list合并图
            graphs = [item['graphs'][graph_type] for item in batch]
            batch_graphs[graph_type] = Batch.from_data_list(graphs)
        
        return {
            'graphs': batch_graphs,
            'labels': labels,
            'teacher_labels': teacher_labels
        }


def create_dataloaders(config):
    """创建训练、验证和测试数据加载器"""
    print("正在创建数据加载器...")
    
    # 创建数据集
    train_dataset = VulnerabilityDataset(
        config.train_path, 
        max_length=config.max_seq_length,
        node_feature_dim=config.node_feature_dim,
        use_codebert=True  # 使用CodeBERT进行向量化
    )
    
    valid_dataset = VulnerabilityDataset(
        config.valid_path, 
        max_length=config.max_seq_length,
        node_feature_dim=config.node_feature_dim,
        is_valid=True,
        use_codebert=True
    )
    
    test_dataset = VulnerabilityDataset(
        config.test_path, 
        max_length=config.max_seq_length,
        node_feature_dim=config.node_feature_dim,
        is_test=True,
        use_codebert=True
    )
    
    # 计算类别权重
    train_labels = np.array([train_dataset.data.iloc[i]['label'] for i in range(len(train_dataset))])
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # 计算类别权重 (反比于样本数量)
    weights = torch.tensor([
        total_samples / (class_counts[0] * len(class_counts)) if 0 in class_counts else 1.0,
        total_samples / (class_counts[1] * len(class_counts)) if 1 in class_counts else 1.0
    ])
    
    print(f"类别分布: {class_counts}")
    print(f"类别权重: {weights}")
    
    # 确定最佳的num_workers数量
    # 在GPU训练时，使用0个worker可以避免一些多进程问题
    # 但在CPU上训练时，使用多个worker可以加速数据加载
    num_workers = 0 if torch.cuda.is_available() else 4
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=VulnerabilityDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False if num_workers == 0 else True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=VulnerabilityDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False if num_workers == 0 else True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=VulnerabilityDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False if num_workers == 0 else True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, valid_loader, test_loader, weights


def process_json_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    处理JSON格式的漏洞检测数据集，分割为训练、验证和测试集

    参数:
    - input_file: 输入的JSON文件路径
    - output_dir: 输出CSV文件的目录路径
    - train_ratio, val_ratio, test_ratio: 数据集分割比例
    - seed: 随机种子
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取JSON数据
    print(f"正在读取数据集: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取代码和标签
    print("正在提取代码和标签...")
    samples = []
    for item in tqdm(data):
        code = item.get('func', '')
        label = item.get('target', 0)  # 使用0作为默认值（无漏洞）
        project = item.get('project', 'unknown')
        commit_id = item.get('commit_id', 'unknown')

        # 数据清洗：移除过长或过短的代码
        if len(code) < 10 or len(code) > 50000:
            continue

        samples.append({
            'code': code,
            'label': label,
            'project': project,
            'commit_id': commit_id
        })

    print(f"总样本数: {len(samples)}")

    # 检查类别分布
    labels = [sample['label'] for sample in samples]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    print(f"漏洞样本数: {positive_count}, 占比: {positive_count / len(labels):.2%}")
    print(f"非漏洞样本数: {negative_count}, 占比: {negative_count / len(labels):.2%}")

    # 分割数据集
    print("正在分割数据集...")
    train_val, test = train_test_split(samples, test_size=test_ratio, random_state=seed, stratify=labels)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=seed,
        stratify=[sample['label'] for sample in train_val]
    )

    # 将分割后的数据集转换为DataFrame
    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)
    test_df = pd.DataFrame(test)

    # 保存为CSV文件
    print("正在保存数据集...")
    train_df[['code', 'label']].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df[['code', 'label']].to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df[['code', 'label']].to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # 保存完整信息（包括项目和commit_id）
    train_df.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'valid_full.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_full.csv'), index=False)

    # 打印数据集分割信息
    print(f"训练集大小: {len(train)}, 验证集大小: {len(val)}, 测试集大小: {len(test)}")
    print(f"数据集已保存至: {output_dir}")

    return {
        'train': train_df,
        'valid': val_df,
        'test': test_df
    }


def balance_dataset(input_dir, output_dir, method='undersample', seed=42):
    """
    平衡数据集，处理类别不平衡问题

    参数:
    - input_dir: 输入CSV文件的目录路径
    - output_dir: 输出平衡后CSV文件的目录路径
    - method: 'undersample'(下采样) 或 'oversample'(上采样)
    - seed: 随机种子
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler

    os.makedirs(output_dir, exist_ok=True)

    # 读取训练集
    train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    X = train_df['code'].values.reshape(-1, 1)
    y = train_df['label'].values

    # 执行采样
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
        print("正在进行下采样...")
    else:
        sampler = RandomOverSampler(random_state=seed)
        print("正在进行上采样...")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # 创建新的DataFrame
    balanced_df = pd.DataFrame({
        'code': X_resampled.flatten(),
        'label': y_resampled
    })

    # 保存平衡后的训练集
    balanced_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    # 复制验证集和测试集
    val_df = pd.read_csv(os.path.join(input_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    val_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # 打印信息
    print(f"原始训练集大小: {len(train_df)}, 正例: {sum(y)}, 负例: {len(y) - sum(y)}")
    print(
        f"平衡后训练集大小: {len(balanced_df)}, 正例: {sum(y_resampled)}, 负例: {len(y_resampled) - sum(y_resampled)}")
    print(f"平衡后的数据集已保存至: {output_dir}")


def analyze_dataset(data_dir):
    """分析数据集的基本统计信息"""
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    print("\n数据集分析:")
    print(f"训练集: {len(train_df)}个样本")
    print(f"  - 漏洞样本: {sum(train_df['label'])}, 占比: {sum(train_df['label']) / len(train_df):.2%}")
    print(
        f"  - 非漏洞样本: {len(train_df) - sum(train_df['label'])}, 占比: {1 - sum(train_df['label']) / len(train_df):.2%}")

    print(f"验证集: {len(val_df)}个样本")
    print(f"  - 漏洞样本: {sum(val_df['label'])}, 占比: {sum(val_df['label']) / len(val_df):.2%}")
    print(f"  - 非漏洞样本: {len(val_df) - sum(val_df['label'])}, 占比: {1 - sum(val_df['label']) / len(val_df):.2%}")

    print(f"测试集: {len(test_df)}个样本")
    print(f"  - 漏洞样本: {sum(test_df['label'])}, 占比: {sum(test_df['label']) / len(test_df):.2%}")
    print(
        f"  - 非漏洞样本: {len(test_df) - sum(test_df['label'])}, 占比: {1 - sum(test_df['label']) / len(test_df):.2%}")

    # 代码长度分析
    train_df['code_length'] = train_df['code'].apply(len)
    print("\n代码长度统计(训练集):")
    print(f"  - 最短: {train_df['code_length'].min()}")
    print(f"  - 最长: {train_df['code_length'].max()}")
    print(f"  - 平均: {train_df['code_length'].mean():.2f}")
    print(f"  - 中位数: {train_df['code_length'].median()}")

    # 漏洞与非漏洞样本的代码长度对比
    vuln_len = train_df[train_df['label'] == 1]['code_length']
    non_vuln_len = train_df[train_df['label'] == 0]['code_length']
    print(f"  - 漏洞样本平均长度: {vuln_len.mean():.2f}")
    print(f"  - 非漏洞样本平均长度: {non_vuln_len.mean():.2f}")


def create_mini_dataset(data_dir, output_dir, samples_per_class=1000):
    """
    创建一个小型数据集用于快速测试

    参数:
    - data_dir: 原始数据集目录
    - output_dir: 输出小型数据集目录
    - samples_per_class: 每个类别的样本数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始数据集
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # 从每个类别中抽取指定数量的样本
    train_vuln = train_df[train_df['label'] == 1].sample(min(samples_per_class, sum(train_df['label'])))
    train_non_vuln = train_df[train_df['label'] == 0].sample(
        min(samples_per_class, len(train_df) - sum(train_df['label'])))

    val_vuln = val_df[val_df['label'] == 1].sample(min(samples_per_class // 5, sum(val_df['label'])))
    val_non_vuln = val_df[val_df['label'] == 0].sample(min(samples_per_class // 5, len(val_df) - sum(val_df['label'])))

    test_vuln = test_df[test_df['label'] == 1].sample(min(samples_per_class // 5, sum(test_df['label'])))
    test_non_vuln = test_df[test_df['label'] == 0].sample(
        min(samples_per_class // 5, len(test_df) - sum(test_df['label'])))

    # 合并并保存
    train_mini = pd.concat([train_vuln, train_non_vuln])
    val_mini = pd.concat([val_vuln, val_non_vuln])
    test_mini = pd.concat([test_vuln, test_non_vuln])

    train_mini.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_mini.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_mini.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"小型数据集创建完成，已保存至: {output_dir}")
    print(f"训练集: {len(train_mini)}个样本，验证集: {len(val_mini)}个样本，测试集: {len(test_mini)}个样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理漏洞检测数据集')
    parser.add_argument('--input', type=str, required=True, help='输入的JSON文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件的目录路径')
    parser.add_argument('--balance', action='store_true', help='是否平衡数据集')
    parser.add_argument('--balance_method', type=str, choices=['undersample', 'oversample'],
                        default='undersample', help='平衡方法: undersample或oversample')
    parser.add_argument('--create_mini', action='store_true', help='是否创建小型数据集用于测试')
    parser.add_argument('--analyze', action='store_true', help='是否分析数据集统计信息')

    args = parser.parse_args()

    # 处理原始JSON数据并分割数据集
    dataset = process_json_dataset(args.input, args.output)

    # 分析数据集
    if args.analyze:
        analyze_dataset(args.output)

    # 平衡数据集
    if args.balance:
        balance_dir = os.path.join(args.output, 'balanced')
        balance_dataset(args.output, balance_dir, method=args.balance_method)
        if args.analyze:
            analyze_dataset(balance_dir)

    # 创建小型数据集用于测试
    if args.create_mini:
        mini_dir = os.path.join(args.output, 'mini')
        create_mini_dataset(args.output, mini_dir)
