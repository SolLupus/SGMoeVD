# config.py
import torch


class Config:
    def __init__(self):
        # 数据相关
        self.train_path = 'processed_data/chromedebain/balanced/train.csv'
        self.valid_path = 'processed_data/chromedebain/balanced/valid.csv'
        self.test_path = 'processed_data/chromedebain/balanced/test.csv'
        self.max_seq_length = 1024
        self.batch_size = 128
        
        # 模型相关
        self.hidden_size = 768
        self.num_classes = 2
        
        # 图神经网络相关
        self.node_feature_dim = 768  # 节点特征维度
        self.gcn_hidden_size = 196   # GCN隐藏层大小
        self.gcn_num_layers = 3      # GCN层数（从8减少到3）
        self.gcn_dropout = 0.1     # GCN dropout

        # 专业化MoE相关参数
        self.num_experts = 8         # 每种图的专家数量
        self.moe_hidden_dim = 512    # MoE隐藏层维度
        self.specialized_moe = True  # 是否使用专业化的MoE
        self.cross_graph_attention = True  # 是否使用跨图注意力
        self.expert_dropout = 0.2    # 专家模型的dropout
        
        # 稀疏门控相关配置ß
        self.use_sparse_moe = True   # 是否使用稀疏门控
        self.sparse_top_k = 1        # 每个样本选择的专家数量
        self.load_balance_coef = 0.01  # 负载均衡损失系数
        
        # 各类图特定专家配置
        self.ast_expert_types = ['tree', 'syntax', 'nested',"light"]  # AST专家类型
        self.pdg_expert_types = ['dataflow', 'dep', 'global',"light"]  # PDG专家类型
        self.cfg_expert_types = ['control', 'path', 'loop',"light"]  # CFG专家类型
        
        # 特征融合相关
        self.fusion_dim = 512        # 融合特征维度
        self.fusion_dropout = 0.2    # 融合层dropout
        
        # 训练相关
        self.num_epochs = 50
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.05
        self.max_grad_norm = 0.5
        self.patience = 8
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存和日志
        self.model_save_path = 'saved_models'
        self.log_dir = 'logs'