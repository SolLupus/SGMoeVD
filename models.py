# models.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, Sequential, GATConv
from torch_geometric.data import Data, Batch


# 使用torch_geometric库的GCN实现
class GCNEncoder(nn.Module):
    """基于PyTorch Geometric的图卷积网络编码器"""
    
    def __init__(self, input_dim, hidden_size, out_features, num_layers=2, dropout=0.2):
        super().__init__()
        self.in_features = input_dim
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 初始投影层 - 支持处理不同维度的节点特征
        # 检测输入是CodeBERT特征(768维)还是手工特征(128维)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
            elif i == num_layers - 1:
                self.gcn_layers.append(GCNConv(hidden_size, out_features))
            else:
                self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
        
        # 规范化和激活
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers-1)])
        self.layer_norms.append(nn.LayerNorm(out_features))
        self.acts = nn.ModuleList([nn.GELU() for _ in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, data):
        """
        前向传播
        data: PyG格式的图数据 (Data或Batch对象)
        """
        # 提取数据
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 检查输入特征维度，如果与预期的in_features不匹配
        # 则需要调整投影层或进行维度转换
        if x.size(-1) != self.in_features:
            # 打印警告信息
            print(f"警告: 输入特征维度 {x.size(-1)} 与预期的维度 {self.in_features} 不匹配")
            
            # 如果是CodeBERT特征，使用平均池化降维
            if x.size(-1) == 768:  # CodeBERT的嵌入维度
                # 创建临时投影层进行降维
                temp_proj = nn.Linear(768, self.in_features).to(x.device)
                x = temp_proj(x)
            else:
                # 其他情况，使用简单的线性投影或截断/填充
                orig_dim = x.size(-1)
                if orig_dim > self.in_features:
                    # 截断到所需维度
                    x = x[:, :self.in_features]
                else:
                    # 填充到所需维度
                    padding = torch.zeros(x.size(0), self.in_features - orig_dim, device=x.device)
                    x = torch.cat([x, padding], dim=1)
        
        # 投影特征
        x = self.input_proj(x)
        
        # 检查和修复边索引 - 确保边索引不超出节点数量
        num_nodes = x.size(0)
        if edge_index.numel() > 0:  # 确保edge_index不为空
            # 检查是否有超出范围的边索引
            mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            if not mask.all():
                # 找到并过滤掉无效的边
                print(f"警告: 发现并移除了 {(~mask).sum().item()} 条无效边 (节点数: {num_nodes})")
                edge_index = edge_index[:, mask]
                
                # 如果所有边都无效，创建自环以避免错误
                if edge_index.size(1) == 0:
                    print("警告: 所有边都无效，创建自环以避免错误")
                    # 为每个节点创建自环
                    self_loops = torch.arange(num_nodes, device=x.device)
                    edge_index = torch.stack([self_loops, self_loops], dim=0)
        
        # GCN层处理
        for i, gcn_layer in enumerate(self.gcn_layers):
            try:
                # 图卷积
                x = gcn_layer(x, edge_index)
                
                # 归一化
                if i < len(self.layer_norms):
                    x = self.layer_norms[i](x)
                
                # 激活函数
                x = self.acts[i](x)
                
                # dropout
                x = self.dropout_layer(x)
            except RuntimeError as e:
                # 捕获运行时错误，打印详细信息，然后重试
                print(f"GCN层 {i} 出错: {str(e)}")
                print(f"节点数: {x.size(0)}, 边数: {edge_index.size(1)}")
                print(f"边索引范围: [{edge_index.min().item()}, {edge_index.max().item()}]")
                
                # 尝试使用稀疏矩阵乘法替代GCN操作
                # 创建一个简单的邻接矩阵
                adj = torch.sparse.FloatTensor(
                    edge_index, 
                    torch.ones(edge_index.size(1), device=edge_index.device),
                    torch.Size([x.size(0), x.size(0)])
                )
                # 简单的图卷积操作
                if i == self.num_layers - 1:
                    x = torch.spmm(adj, x) @ torch.randn(x.size(1), self.out_features, device=x.device)
                else:
                    x = torch.spmm(adj, x)
                
                # 应用激活和规范化
                if i < len(self.layer_norms):
                    x = self.layer_norms[i](x)
                x = self.acts[i](x)
                x = self.dropout_layer(x)
        
        # 全局池化
        if batch is None:
            # 如果没有batch信息，使用平均池化
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_max = torch.max(x, dim=0, keepdim=True)[0]
        else:
            try:
                x_mean = global_mean_pool(x, batch)  # [batch_size, out_features]
                x_max = global_max_pool(x, batch)    # [batch_size, out_features]
            except RuntimeError as e:
                print(f"全局池化出错: {str(e)}")
                # 创建一个安全的batch张量，确保其中的索引不超过节点数量
                safe_batch = torch.clamp(batch, 0, batch.max().item())
                # 使用安全的batch进行池化
                x_mean = global_mean_pool(x, safe_batch)
                x_max = global_max_pool(x, safe_batch)
        
        # 合并两种池化
        pooled = x_mean + x_max  # [batch_size, out_features]
        
        return pooled


class EnhancedVulnerabilityDetector(nn.Module):
    """基于多图GCN的漏洞检测模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 应用权重衰减（L2正则化）用于防止过拟合
        self.weight_decay = config.weight_decay

        self.pdg_encoder = GCNEncoder(
            input_dim=config.node_feature_dim,
            hidden_size=config.gcn_hidden_size,
            out_features=512,
            num_layers=config.gcn_num_layers,
            dropout=config.gcn_dropout
        )
        
        # CPG (Control Flow Graph)
        self.cpg_encoder = GCNEncoder(
            input_dim=config.node_feature_dim,
            hidden_size=config.gcn_hidden_size,
            out_features=512,
            num_layers=config.gcn_num_layers,
            dropout=config.gcn_dropout
        )
        
        # AST (Abstract Syntax Tree)
        self.ast_encoder = GCNEncoder(
            input_dim=config.node_feature_dim,
            hidden_size=config.gcn_hidden_size,
            out_features=512,
            num_layers=config.gcn_num_layers,
            dropout=config.gcn_dropout
        )
        
        # 使用专业化的图混合专家模型处理三种图特征
        self.graph_moe = SpecializedGraphMixtureOfExperts(
            graph_dim=512,
            num_experts=config.num_experts,
            hidden_dim=512,
            output_dim=config.fusion_dim,
            dropout=config.fusion_dropout,
            sparse_top_k=config.sparse_top_k
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, config.num_classes)
        )
        
        # 初始化网络权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, module in self.named_modules():
            # 跳过预训练模型
            if "coder_model" in name:
                continue
                
            # 初始化线性层
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            # 初始化LayerNorm
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, graphs):
        """
        前向传播
        graphs: 包含pdg, cpg, ast三种图
        """
        
        # 1. 提取三种图的特征
        pdg_features = self.pdg_encoder(graphs['pdg'])  # [batch_size, 512]
        cpg_features = self.cpg_encoder(graphs['cpg'])  # [batch_size, 512]
        ast_features = self.ast_encoder(graphs['ast'])  # [batch_size, 512]
        
        # 2. 将三种图特征直接送入图混合专家模型
        graph_features_list = [pdg_features, cpg_features, ast_features]
        fused_features, gate_probs, expert_usage_stats = self.graph_moe(graph_features_list)  # [batch_size, fusion_dim]
        
        # 3. 分类
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        return logits, gate_probs, expert_usage_stats

# 专业化的图混合专家模型 - 针对AST、CFG和PDG三种不同类型的图
class SpecializedGraphMixtureOfExperts(nn.Module):
    def __init__(self, graph_dim, num_experts=4, hidden_dim=512, output_dim=512, dropout=0.2,sparse_top_k=2):
        super().__init__()
        self.graph_dim = graph_dim
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.sparse_top_k = sparse_top_k
        
        # 计算每个图专家输出的维度，确保总和等于output_dim
        self.per_graph_dim = (output_dim // 3) // 4 * 4  # 确保per_graph_dim是4的倍数
        # 确保能被3整除
        self.final_output_dim = self.per_graph_dim * 3
        
        # ===== AST专家 - 专注于树结构和语法特征 =====
        self.ast_experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 4 == 0:
                # 树结构特征专家 - 使用多层感知机捕获节点间层级关系
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            elif i % 4 == 1:
                # 语法特征专家 - 提取语法元素信息
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    # 使用注意力机制捕获局部结构
                    nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
                # 修改forward方法以处理多头注意力层
                expert_forward = expert.forward
                def new_forward(x):
                    for module_idx, module in enumerate(expert):
                        if isinstance(module, nn.MultiheadAttention):
                            # 对输入进行reshape以适应注意力机制
                            batch_size = x.size(0)
                            x = x.view(batch_size, 1, -1)  # [batch_size, 1, hidden_dim]
                            x, _ = module(x, x, x)
                            x = x.squeeze(1)  # [batch_size, hidden_dim]
                        else:
                            x = module(x)
                    return x
                expert.forward = new_forward
            elif i % 4 == 2:
                # 嵌套结构专家 - 处理嵌套代码块
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    # 使用不同的激活函数捕获非线性关系
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            else:
                # 轻量级专家 - 快速直接处理
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim//2),
                    nn.LayerNorm(hidden_dim//2),
                    nn.GELU(),
                    nn.Dropout(dropout/2),  # 较少的dropout
                    nn.Linear(hidden_dim//2, self.per_graph_dim)
                )
            
            self.ast_experts.append(expert)
            
        # ===== PDG专家 - 专注于数据流和依赖关系 =====
        self.pdg_experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 4 == 0:
                # 数据流专家 - 捕获变量间的数据流动关系
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    # 残差连接 - 保持原始数据流信息
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            elif i % 4 == 1:
                # 变量依赖专家 - 强调变量间依赖关系
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Sigmoid(),  # 使用Sigmoid强调依赖性强度
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            elif i % 4 == 2:
                # 全局依赖专家 - 从整体视角捕获依赖
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim*2),  # 更宽的网络
                    nn.LayerNorm(hidden_dim*2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim*2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            else:
                # 轻量级依赖专家 - 快速简单处理
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim//2),
                    nn.LayerNorm(hidden_dim//2),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim//2, self.per_graph_dim)
                )
            
            self.pdg_experts.append(expert)
            
        # ===== CFG专家 - 专注于控制流和执行路径 =====
        self.cfg_experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 4 == 0:
                # 控制流专家 - 强调控制流的顺序和跳转
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    # 使用较深的网络捕获复杂的控制流
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            elif i % 4 == 1:
                # 路径特征专家 - 专注于执行路径
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),  # 使用Tanh捕获路径依赖性的正负关系
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            elif i % 4 == 2:
                # 循环嵌套专家 - 处理循环和嵌套结构
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    # 门控机制 - 控制信息流动
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim*2),
                        nn.Sigmoid(),
                        nn.Linear(hidden_dim*2, hidden_dim),
                    ),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.per_graph_dim)
                )
            else:
                # 轻量级CFG专家 - 处理简单控制流
                expert = nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim//2),
                    nn.ReLU(),
                    nn.Dropout(dropout/2),
                    nn.Linear(hidden_dim//2, self.per_graph_dim)
                )
            
            self.cfg_experts.append(expert)
        
        # 专业化的门控网络 - 为每种图类型选择最佳专家
        # AST门控 - 检测语法特征
        self.ast_gate = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_experts)
        )
        
        # PDG门控 - 检测数据流特征
        self.pdg_gate = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_experts)
        )
        
        # CFG门控 - 检测控制流特征
        self.cfg_gate = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_experts)
        )
        
        # 图交互层 - 学习图间关系
        self.graph_interaction = nn.Sequential(
            nn.Linear(self.final_output_dim, self.final_output_dim),
            nn.LayerNorm(self.final_output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.final_output_dim, self.final_output_dim),
            nn.LayerNorm(self.final_output_dim),
            nn.Sigmoid()  # 控制每个特征的重要性
        )
        
        # 图间注意力 - 学习不同图之间的关系
        self.cross_graph_attention = nn.MultiheadAttention(
            embed_dim=self.per_graph_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出变换层
        self.output_transform = nn.Sequential(
            nn.Linear(self.final_output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化专家权重"""
        # 为不同类型的图使用不同的初始化策略
        
        # AST专家 - 语法树结构专家初始化
        for i, expert in enumerate(self.ast_experts):
            for name, module in expert.named_modules():
                if isinstance(module, nn.Linear):
                    # 使用Xavier均匀初始化
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # PDG专家 - 数据流依赖专家初始化
        for i, expert in enumerate(self.pdg_experts):
            for name, module in expert.named_modules():
                if isinstance(module, nn.Linear):
                    # 使用He初始化 - 适合ReLU激活函数
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # CFG专家 - 控制流专家初始化
        for i, expert in enumerate(self.cfg_experts):
            for name, module in expert.named_modules():
                if isinstance(module, nn.Linear):
                    # 使用正态分布初始化 - 适合Tanh和Sigmoid激活
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, graph_features_list):
        """
        前向传播
        graph_features_list: 图特征列表 [pdg_features, cpg_features, ast_features]
        每个图特征的形状: [batch_size, graph_dim]
        """
        batch_size = graph_features_list[0].size(0)
        
        # 解包三种图特征
        pdg_features, cpg_features, ast_features = graph_features_list
        
        # 稀疏门控参数
        k = self.sparse_top_k  # 每个样本选择top-k个专家
        
        # 处理AST特征 - 使用稀疏门控
        ast_gate_logits = self.ast_gate(ast_features)
        # 选择top-k个专家
        ast_top_k_logits, ast_indices = torch.topk(ast_gate_logits, k=k, dim=-1)
        # 对选中的logits应用softmax
        ast_top_k_probs = F.softmax(ast_top_k_logits, dim=-1)
        # 创建稀疏门控权重
        ast_gate_probs = torch.zeros_like(ast_gate_logits).scatter_(-1, ast_indices, ast_top_k_probs)
        
        ast_expert_outputs = []
        for expert in self.ast_experts:
            ast_expert_outputs.append(expert(ast_features))
        
        stacked_ast_outputs = torch.stack(ast_expert_outputs, dim=1)  # [batch_size, num_experts, per_graph_dim]
        ast_output = (ast_gate_probs.unsqueeze(-1) * stacked_ast_outputs).sum(dim=1)  # [batch_size, per_graph_dim]
        
        # 处理PDG特征 - 使用稀疏门控
        pdg_gate_logits = self.pdg_gate(pdg_features)
        # 选择top-k个专家
        pdg_top_k_logits, pdg_indices = torch.topk(pdg_gate_logits, k=k, dim=-1)
        # 对选中的logits应用softmax
        pdg_top_k_probs = F.softmax(pdg_top_k_logits, dim=-1)
        # 创建稀疏门控权重
        pdg_gate_probs = torch.zeros_like(pdg_gate_logits).scatter_(-1, pdg_indices, pdg_top_k_probs)
        
        pdg_expert_outputs = []
        for expert in self.pdg_experts:
            pdg_expert_outputs.append(expert(pdg_features))
        
        stacked_pdg_outputs = torch.stack(pdg_expert_outputs, dim=1)
        pdg_output = (pdg_gate_probs.unsqueeze(-1) * stacked_pdg_outputs).sum(dim=1)
        
        # 处理CFG特征 - 使用稀疏门控
        cfg_gate_logits = self.cfg_gate(cpg_features)
        # 选择top-k个专家
        cfg_top_k_logits, cfg_indices = torch.topk(cfg_gate_logits, k=k, dim=-1)
        # 对选中的logits应用softmax
        cfg_top_k_probs = F.softmax(cfg_top_k_logits, dim=-1)
        # 创建稀疏门控权重
        cfg_gate_probs = torch.zeros_like(cfg_gate_logits).scatter_(-1, cfg_indices, cfg_top_k_probs)
        
        cfg_expert_outputs = []
        for expert in self.cfg_experts:
            cfg_expert_outputs.append(expert(cpg_features))
        
        stacked_cfg_outputs = torch.stack(cfg_expert_outputs, dim=1)
        cfg_output = (cfg_gate_probs.unsqueeze(-1) * stacked_cfg_outputs).sum(dim=1)
        
        # 图间信息交互 - 使用自注意力机制使三种图相互了解
        graph_outputs = torch.stack([pdg_output, cfg_output, ast_output], dim=1)  # [batch_size, 3, per_graph_dim]
        
        # 应用跨图注意力 - 让三种图之间相互交流
        cross_attn_output, _ = self.cross_graph_attention(
            graph_outputs, graph_outputs, graph_outputs
        )  # [batch_size, 3, per_graph_dim]
        
        # 残差连接
        enhanced_outputs = graph_outputs + cross_attn_output  # [batch_size, 3, per_graph_dim]
        
        # 拉平输出
        enhanced_pdg_output = enhanced_outputs[:, 0]  # [batch_size, per_graph_dim]
        enhanced_cfg_output = enhanced_outputs[:, 1]  # [batch_size, per_graph_dim]
        enhanced_ast_output = enhanced_outputs[:, 2]  # [batch_size, per_graph_dim]
        
        # 拼接所有图的输出
        combined_output = torch.cat([enhanced_pdg_output, enhanced_cfg_output, enhanced_ast_output], dim=-1)  # [batch_size, final_output_dim]
        
        # 应用图交互注意力 - 学习特征重要性
        interaction_weights = self.graph_interaction(combined_output)
        attended_output = combined_output * interaction_weights
        
        # 最终输出变换
        final_output = self.output_transform(attended_output)  # [batch_size, output_dim]
        
        # 合并所有专家的选择权重，用于可视化和分析
        all_gate_probs = torch.cat([
            pdg_gate_probs, cfg_gate_probs, ast_gate_probs
        ], dim=-1)  # [batch_size, num_experts*3]
        
        # 计算专家使用统计 - 对于监控专家使用情况
        pdg_expert_usage = pdg_gate_probs.sum(dim=0) / batch_size  # 每个PDG专家的平均使用率
        cfg_expert_usage = cfg_gate_probs.sum(dim=0) / batch_size  # 每个CFG专家的平均使用率
        ast_expert_usage = ast_gate_probs.sum(dim=0) / batch_size  # 每个AST专家的平均使用率
        
        # 结合所有专家使用统计，用于监控
        expert_usage_stats = {
            'pdg': pdg_expert_usage,
            'cfg': cfg_expert_usage,
            'ast': ast_expert_usage
        }
        
        return final_output, all_gate_probs, expert_usage_stats
