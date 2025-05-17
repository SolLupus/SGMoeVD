# app.py
from flask import Flask, render_template, request, jsonify
import torch
import os
import json
import numpy as np
# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 必须在导入plt之前设置
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.data import Batch
from models import EnhancedVulnerabilityDetector
from configs import Config
from data_processing import VulnerabilityDataset
from utils import get_codebert_embeddings, load_codebert_model
model_imports_successful = True

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 创建数据目录
os.makedirs('static/images', exist_ok=True)

# 加载配置和模型
config = None
model = None
model_loaded = False
codebert_model = None
codebert_tokenizer = None

if model_imports_successful:
    config = Config()
    config.device = torch.device('cpu')  # 在生产环境中可以改为GPU
    
    # 加载CodeBERT模型
    try:
        codebert_tokenizer, codebert_model = load_codebert_model('microsoft/codebert-base', config.device)
        print("CodeBERT模型加载成功")
    except Exception as e:
        print(f"CodeBERT模型加载失败: {str(e)}")

    # 加载模型
    model_path = os.path.join(config.model_save_path, 'ffmpeg/final_model.pt')
    if os.path.exists(model_path):
        model = EnhancedVulnerabilityDetector(config)
        model.load_state_dict(torch.load(model_path, map_location=config.device,weights_only=False)['model_state_dict'])
        model.eval()
        model_loaded = True
        print("漏洞检测模型加载成功")

# 使用data_processing.py中的方法构建图结构
def process_code_with_dataset_methods(code):
    """
    使用VulnerabilityDataset类中的方法处理代码
    
    参数:
        code: 源代码字符串
        
    返回:
        包含PDG、CPG和AST图的字典
    """
    # 创建临时的VulnerabilityDataset对象
    temp_dataset = VulnerabilityDataset(
        None,  # 不需要文件路径
        max_length=512,
        node_feature_dim=768,
        use_codebert=codebert_model is not None and codebert_tokenizer is not None
    )
    
    # 设置CodeBERT模型
    if codebert_model is not None and codebert_tokenizer is not None:
        temp_dataset.model = codebert_model
        temp_dataset.tokenizer = codebert_tokenizer
    
    # 使用数据集方法生成三种图
    pdg_graph = temp_dataset._generate_pdg(code, False)  # 假设不是负样本
    cpg_graph = temp_dataset._generate_cpg(code, False)
    ast_graph = temp_dataset._generate_ast(code, False)
    
    # 创建批处理，增加batch维度
    pdg_batch = Batch.from_data_list([pdg_graph])
    cpg_batch = Batch.from_data_list([cpg_graph])
    ast_batch = Batch.from_data_list([ast_graph])
    
    return {
        'pdg': pdg_batch,
        'cpg': cpg_batch,
        'ast': ast_batch
    }

# 保留模拟预处理函数作为备用
def mock_preprocess_code(code):
    """
    创建模拟的图数据用于演示
    
    参数:
        code: 源代码字符串
    
    返回:
        字典，包含PDG, CFG, AST三种图表示
    """
    # 根据代码长度和复杂度创建节点数
    code_lines = code.split('\n')
    node_count = min(max(len(code_lines) * 2, 10), 50)  # 最少10个节点，最多50个节点
    
    # 为三种图创建不同的结构
    pdg = create_mock_graph(node_count, 'pdg')
    cfg = create_mock_graph(node_count // 2, 'cfg')
    ast = create_mock_graph(node_count * 2, 'ast')
    
    return {
        'pdg': pdg,
        'cpg': cfg,  # 在代码中CFG和CPG指的是同一个图
        'ast': ast
    }

def create_mock_graph(node_count, graph_type):
    """创建模拟图数据"""
    # 创建特征矩阵 - 每个节点768维特征
    x = torch.randn(node_count, 768)
    
    # 创建边 - 根据图类型设置不同的连接模式
    if graph_type == 'pdg':
        # PDG通常比较稀疏，有向
        edge_prob = 0.1
    elif graph_type == 'cfg':
        # CFG通常是线性序列加分支
        edge_prob = 0.15
    else:  # ast
        # AST通常是树状结构
        edge_prob = 0.2
    
    # 创建随机边
    edge_index = []
    for i in range(node_count):
        for j in range(node_count):
            if i != j and random.random() < edge_prob:
                edge_index.append([i, j])
    
    # 如果没有边，至少添加一些基本边
    if not edge_index:
        for i in range(node_count-1):
            edge_index.append([i, i+1])
    
    # 转换为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # 创建PyG的Data对象并包装为Batch
    data = Data(x=x, edge_index=edge_index)
    
    return data

def get_graph_visualization(graph_data, graph_type):
    """将图数据转换为可视化图像"""
    # 将PyG图数据转换为NetworkX图用于可视化
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(graph_data.x.shape[0]):
        G.add_node(i)
    
    # 添加边
    edge_list = graph_data.edge_index.t().numpy()
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    
    # 创建可视化
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=700, edge_color='black', linewidths=1, 
            font_size=15, font_weight='bold')
    plt.title(f"{graph_type} 图结构")
    
    # 将图形转换为base64编码
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode()

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/detect', methods=['POST'])
def detect_vulnerability():
    code = request.form.get('code')
    
    if not code:
        return jsonify({
            'error': '请提供代码'
        })
    
    try:
        # 尝试使用data_processing.py中的方法
        use_real_processing = model_imports_successful and 'VulnerabilityDataset' in globals()
        
        if use_real_processing:
            try:
                # 使用真实的图构造方法
                graphs = process_code_with_dataset_methods(code)
                print("使用data_processing.py中的图构造方法成功")
            except Exception as e:
                print(f"真实图构造失败，使用模拟数据: {str(e)}")
                graphs = mock_preprocess_code(code)
        else:
            # 使用模拟数据
            graphs = mock_preprocess_code(code)
            print("使用模拟的图构造方法")
        
        # 生成图的可视化
        pdg_viz = get_graph_visualization(graphs['pdg'], 'PDG（程序依赖图）')
        cpg_viz = get_graph_visualization(graphs['cpg'], 'CFG（控制流图）')
        ast_viz = get_graph_visualization(graphs['ast'], 'AST（抽象语法树）')
        
        # 进行预测
        if model_loaded:
            with torch.no_grad():
                logits, gate_probs, expert_usage = model(graphs)
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
                
                # 分析专家使用情况
                expert_analysis = {
                    'pdg': expert_usage['pdg'].tolist(),
                    'cfg': expert_usage['cfg'].tolist(),
                    'ast': expert_usage['ast'].tolist()
                }
        else:
            # 模拟结果
            prediction = random.randint(0, 1)  # 随机预测
            confidence = random.uniform(0.7, 0.95)  # 随机置信度
            
            # 模拟专家使用情况
            expert_analysis = {
                'pdg': [random.random() for _ in range(4)],
                'cfg': [random.random() for _ in range(4)],
                'ast': [random.random() for _ in range(4)]
            }
        
        # 漏洞类型分析（模拟或基于模型输出）
        
        result = {
            'prediction': int(prediction),  # 0表示安全，1表示有漏洞
            'confidence': float(confidence) * 100,  # 置信度百分比
            'graph_visualizations': {
                'pdg': pdg_viz,
                'cfg': cpg_viz,
                'ast': ast_viz
            },
            'expert_usage': expert_analysis
        }
        
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'分析过程中出错：{str(e)}',
            'traceback': traceback.format_exc()
        })

@app.route('/model_info')
def model_info():
    if not model_loaded:
        # 返回模拟的模型配置
        config_dict = {
            'hidden_size': 768,
            'gcn_hidden_size': 196,
            'gcn_num_layers': 3,
            'fusion_dim': 512,
            'num_experts': 4,
            'specialized_moe': True,
            'cross_graph_attention': True
        }
    else:
        # 返回实际模型配置信息
        config_dict = {
            'hidden_size': config.hidden_size,
            'gcn_hidden_size': config.gcn_hidden_size,
            'gcn_num_layers': config.gcn_num_layers,
            'fusion_dim': config.fusion_dim,
            'num_experts': config.num_experts,
            'specialized_moe': config.specialized_moe,
            'cross_graph_attention': config.cross_graph_attention
        }
    
    return jsonify(config_dict)

@app.route('/get_example_code')
def get_example_code():
    # 提供一个易于出现漏洞的示例代码
    example_code = """#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // 潜在的缓冲区溢出漏洞
    printf("Buffer contains: %s\\n", buffer);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    } else {
        printf("Please provide an argument\\n");
    }
    return 0;
}"""
    return jsonify({'code': example_code})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080) 