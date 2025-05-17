# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from sklearn.metrics import confusion_matrix
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示成方块的问题

def evaluate_model(model, dataloader, device, print_details=False):
    """
    评估模型在给定数据集上的性能
    
    参数:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 计算设备
        print_details: 是否打印详细评估信息
    
    返回:
        包含各种评估指标的字典
    """
    # 切换到评估模式
    model.eval()
    
    # 存储所有预测和标签
    all_preds = []
    all_labels = []
    
    # 用于计算每个类别的准确率
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    # 存储各种指标
    clean_losses = []
    
    # 禁用梯度计算
    with torch.no_grad():
        # 遍历数据集
        for batch in dataloader:
            # 提取图数据
            graphs = {
                'pdg': batch['graphs']['pdg'].to(device),
                'cpg': batch['graphs']['cpg'].to(device),
                'ast': batch['graphs']['ast'].to(device)
            }
            
            # 提取标签
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(graphs)
            
            # 处理模型输出 - 支持不同的输出格式
            if isinstance(outputs, tuple):
                logits = outputs[0]  # 稀疏门控模型返回(logits, gate_probs, expert_usage_stats)
            else:
                logits = outputs  # 普通模型直接返回logits
            
            # 计算损失
            loss = F.cross_entropy(logits, labels, reduction='mean')
            clean_losses.append(loss.item())
            
            # 获取预测
            preds = torch.argmax(logits, dim=1)
            
            # 收集预测和标签用于计算整体指标
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 计算每个类别的准确率
            for c in [0, 1]:
                mask = (labels == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == labels) & mask).sum().item()
    
    # 计算指标
    metrics = {}
    
    # 准确率
    accuracy = accuracy_score(all_labels, all_preds)
    metrics['accuracy'] = accuracy
    
    # 精确率、召回率、F1值和混淆矩阵
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['confusion_matrix'] = conf_matrix.tolist()
    
    # 每个类别的准确率
    for c in [0, 1]:
        class_acc = class_correct[c] / max(class_total[c], 1)
        metrics[f'class_{c}_accuracy'] = class_acc
    
    # 平均损失
    avg_loss = np.mean(clean_losses)
    metrics['loss'] = avg_loss
    
    # 打印详细评估信息
    if print_details:
        print("\n===== 评估指标 =====")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1值: {f1:.4f}")
        print(f"平均损失: {avg_loss:.4f}")
        print(f"负类准确率: {metrics['class_0_accuracy']:.4f} ({class_correct[0]}/{class_total[0]})")
        print(f"正类准确率: {metrics['class_1_accuracy']:.4f} ({class_correct[1]}/{class_total[1]})")
        print("\n混淆矩阵:")
        print(conf_matrix)
        print("---------------------\n")
    
    return metrics


def plot_training_metrics(train_losses, train_metrics_history, val_metrics_history, save_path, ce_losses=None):
    """
    绘制训练过程中的各种指标
    
    参数:
        train_losses: 训练损失列表
        train_metrics_history: 训练集指标历史
        val_metrics_history: 验证集指标历史
        save_path: 图表保存路径
        ce_losses: 交叉熵损失列表（可选）
    """
    plt.style.use('ggplot')
    
    # 创建一个大图表
    fig = plt.figure(figsize=(15, 20))
    grid = plt.GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 训练损失图表
    ax_loss = fig.add_subplot(grid[0, :])
    ax_loss.plot(train_losses, 'b-', label='Total Loss', linewidth=2)
    
    # 如果提供了CE损失和KD损失，也将它们绘制在同一个子图中
    # if ce_losses is not None:
    #     ax_loss.plot(ce_losses, 'r-', label='CE loss', linewidth=2)
    
    ax_loss.set_xlabel('epoch', fontsize=12)
    ax_loss.set_ylabel('loss', fontsize=12)
    ax_loss.set_title('Loss Change', fontsize=14, fontweight='bold')
    ax_loss.legend(fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    
    axtrain_acc_f1 = fig.add_subplot(grid[1, 0])
    

    train_acc = [m['accuracy'] for m in train_metrics_history]
    train_f1 = [m['f1'] for m in train_metrics_history]
    
    axtrain_acc_f1.plot(train_acc, 'b-', label='Train Accuracy', linewidth=1, marker='o')
    axtrain_acc_f1.plot(train_f1, 'r-', label='Train F1 score', linewidth=1, marker='s')
    axtrain_acc_f1.set_xlabel('epoch', fontsize=12)
    axtrain_acc_f1.set_ylabel('score', fontsize=12)
    axtrain_acc_f1.set_title('Train Accuracy and F1 score', fontsize=14, fontweight='bold')
    axtrain_acc_f1.legend(fontsize=12)
    axtrain_acc_f1.grid(True, linestyle='--', alpha=0.7)
    
    axtrain_prec_rec = fig.add_subplot(grid[1, 1])
    
    train_precision = [m['precision'] for m in train_metrics_history]
    train_recall = [m['recall'] for m in train_metrics_history]
    
    axtrain_prec_rec.plot(train_precision, 'g-', label='Train Precision', linewidth=1, marker='o')
    axtrain_prec_rec.plot(train_recall, 'm-', label='Train Recall', linewidth=1, marker='s')
    axtrain_prec_rec.set_xlabel('epoch', fontsize=12)
    axtrain_prec_rec.set_ylabel('score', fontsize=12)
    axtrain_prec_rec.set_title('Train Precision and Recall', fontsize=14, fontweight='bold')
    axtrain_prec_rec.legend(fontsize=12)
    axtrain_prec_rec.grid(True, linestyle='--', alpha=0.7)
    
    
    
    # 2. 验证集准确率和F1分数
    ax_acc_f1 = fig.add_subplot(grid[2, 0])
    
    # 提取验证指标
    val_acc = [m['accuracy'] for m in val_metrics_history]
    val_f1 = [m['f1'] for m in val_metrics_history]
    
    ax_acc_f1.plot(val_acc, 'b-', label='Validate Accuracy', linewidth=2, marker='o')
    ax_acc_f1.plot(val_f1, 'r-', label='Validate F1 score', linewidth=2, marker='s')
    ax_acc_f1.set_xlabel('epoch', fontsize=12)
    ax_acc_f1.set_ylabel('score', fontsize=12)
    ax_acc_f1.set_title('Validate Accuracy and F1 score', fontsize=14, fontweight='bold')
    ax_acc_f1.legend(fontsize=12)
    ax_acc_f1.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 验证集精确率和召回率
    ax_prec_rec = fig.add_subplot(grid[2, 1])
    
    val_precision = [m['precision'] for m in val_metrics_history]
    val_recall = [m['recall'] for m in val_metrics_history]
    
    ax_prec_rec.plot(val_precision, 'g-', label='Validate Precision', linewidth=2, marker='o')
    ax_prec_rec.plot(val_recall, 'm-', label='Validate Recall', linewidth=2, marker='s')
    ax_prec_rec.set_xlabel('epoch', fontsize=12)
    ax_prec_rec.set_ylabel('score', fontsize=12)
    ax_prec_rec.set_title('Precision and Recall', fontsize=14, fontweight='bold')
    ax_prec_rec.legend(fontsize=12)
    ax_prec_rec.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 类别级别精确率对比
    ax_class_prec = fig.add_subplot(grid[3, 0])
    
    c0_precision = [m.get('class_0_precision', 0) for m in val_metrics_history]
    c1_precision = [m.get('class_1_precision', 0) for m in val_metrics_history]
    
    ax_class_prec.plot(c0_precision, 'c-', label='Precision 0', linewidth=2, marker='o')
    ax_class_prec.plot(c1_precision, 'y-', label='Precision 1', linewidth=2, marker='s')
    ax_class_prec.set_xlabel('epoch', fontsize=12)
    ax_class_prec.set_ylabel('score', fontsize=12)
    ax_class_prec.set_title('Precision compare', fontsize=14, fontweight='bold')
    ax_class_prec.legend(fontsize=12)
    ax_class_prec.grid(True, linestyle='--', alpha=0.7)
    
    # 5. 类别级别召回率对比
    ax_class_rec = fig.add_subplot(grid[3, 1])
    
    c0_recall = [m.get('class_0_recall', 0) for m in val_metrics_history]
    c1_recall = [m.get('class_1_recall', 0) for m in val_metrics_history]
    
    ax_class_rec.plot(c0_recall, 'c--', label='recall 0', linewidth=2, marker='o')
    ax_class_rec.plot(c1_recall, 'y--', label='recall 1', linewidth=2, marker='s')
    ax_class_rec.set_xlabel('epoch', fontsize=12)
    ax_class_rec.set_ylabel('recall', fontsize=12)
    ax_class_rec.set_title('Recall compare', fontsize=14, fontweight='bold')
    ax_class_rec.legend(fontsize=12)
    ax_class_rec.grid(True, linestyle='--', alpha=0.7)
    
    # 6. 类别级别F1分数对比
    ax_class_f1 = fig.add_subplot(grid[4, 0])
    
    c0_f1 = [m.get('class_0_f1', 0) for m in val_metrics_history]
    c1_f1 = [m.get('class_1_f1', 0) for m in val_metrics_history]
    
    ax_class_f1.plot(c0_f1, 'c-.', label='F1 0', linewidth=2, marker='o')
    ax_class_f1.plot(c1_f1, 'y-.', label='F1 1', linewidth=2, marker='s')
    ax_class_f1.set_xlabel('epoch', fontsize=12)
    ax_class_f1.set_ylabel('F1score', fontsize=12)
    ax_class_f1.set_title('F1 compare', fontsize=14, fontweight='bold')
    ax_class_f1.legend(fontsize=12)
    ax_class_f1.grid(True, linestyle='--', alpha=0.7)
    
    # 7. ROC AUC和PR AUC
    ax_auc = fig.add_subplot(grid[4, 1])
    
    roc_auc = [m.get('roc_auc', 0) for m in val_metrics_history]
    pr_auc = [m.get('pr_auc', 0) for m in val_metrics_history]
    
    ax_auc.plot(roc_auc, 'b-.', label='ROC AUC', linewidth=2, marker='o')
    ax_auc.plot(pr_auc, 'r-.', label='PR AUC', linewidth=2, marker='s')
    ax_auc.set_xlabel('epoch', fontsize=12)
    ax_auc.set_ylabel('AUC score', fontsize=12)
    ax_auc.set_title('ROC AUC and PR AUC', fontsize=14, fontweight='bold')
    ax_auc.legend(fontsize=12)
    ax_auc.grid(True, linestyle='--', alpha=0.7)
    
    # 添加总标题
    fig.suptitle('Model trains\' and validations\' metrics', fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, scheduler, load_path):
    """加载检查点"""
    checkpoint = torch.load(load_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']

def save_metrics(metrics, save_path):
    """保存评估指标"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(load_path):
    """加载评估指标"""
    with open(load_path, 'r') as f:
        return json.load(f)

def create_directory(path):
    """创建目录，如果目录不存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_codebert_embeddings(code, tokenizer, model, cache_dir='codebert_cache', max_length=512, use_cache=False):
    """
    获取代码的CodeBERT嵌入，并支持本地缓存
    
    参数:
        code: 代码字符串
        tokenizer: CodeBERT tokenizer
        model: CodeBERT model
        cache_dir: 缓存目录
        max_length: 最大序列长度
        use_cache: 是否使用缓存
    
    返回:
        token_embeddings: 代码的token级嵌入 [num_tokens, embedding_dim]
    """
    import os
    import pickle
    import hashlib
    import torch
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    
    # 如果启用缓存
    if use_cache:
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成唯一的缓存文件名（基于代码内容的哈希）
        code_hash = hashlib.md5(code.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{code_hash}.pkl")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                    # 将加载的嵌入转移到正确的设备上
                    return embeddings.to(device)
            except Exception as e:
                print(f"读取缓存失败: {e}")
    
    # 对代码进行tokenize
    inputs = tokenizer(
        code, 
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 移动到模型所在设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 计算嵌入，使用GPU加速
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 使用最后一层隐藏状态作为token嵌入
        hidden_states = outputs.hidden_states[-1]
        
        # 去除批次维度
        token_embeddings = hidden_states.squeeze(0)
    
    # 如果启用缓存，保存到缓存
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                # 保存时转到CPU以节省内存
                pickle.dump(token_embeddings.cpu(), f)
        except Exception as e:
            print(f"写入缓存失败: {e}")
    
    # 返回嵌入值，保持在相同设备上
    return token_embeddings

def load_codebert_model(model_name='microsoft/codebert-base', device='cuda'):
    """
    加载CodeBERT模型和分词器
    
    参数:
        model_name: 模型名称或路径
        device: 运行设备
    
    返回:
        tokenizer: CodeBERT tokenizer
        model: CodeBERT model
    """
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    # 如果没有GPU，则使用CPU
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # 设置为评估模式
    
    return tokenizer, model

def analyze_moe_experts(model, dataloader, device, save_path=None):
    """
    分析模型中的MoE专家选择情况，并生成可视化结果
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 运行设备
        save_path: 结果保存路径
    
    Returns:
        experts_stats: 专家使用统计信息
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict
    import torch.nn.functional as F
    
    # 设置模型为评估模式
    model.eval()
    
    # 统计不同图类型的专家激活情况
    pdg_expert_weights = []
    cfg_expert_weights = []
    ast_expert_weights = []
    
    # 每个批次收集专家权重
    with torch.no_grad():
        for batch in dataloader:
            # 处理图数据
            graphs = {
                'pdg': batch['graphs']['pdg'].to(device),
                'cpg': batch['graphs']['cpg'].to(device),
                'ast': batch['graphs']['ast'].to(device)
            }
            
            # 提取特征
            pdg_features = model.pdg_encoder(graphs['pdg'])
            cpg_features = model.cpg_encoder(graphs['cpg'])
            ast_features = model.ast_encoder(graphs['ast'])
            
            # 获取专家选择权重
            if hasattr(model.graph_moe, 'pdg_gate'):
                # 专业化MoE模型
                # 使用和前向传播相同的专家选择逻辑
                pdg_gate_logits = model.graph_moe.pdg_gate(pdg_features)
                cfg_gate_logits = model.graph_moe.cfg_gate(cpg_features)
                ast_gate_logits = model.graph_moe.ast_gate(ast_features)
                
                # 检查是否使用稀疏门控
                if hasattr(model.config, 'use_sparse_moe') and model.config.use_sparse_moe:
                    # 稀疏门控 - 使用top-k选择
                    k = 2  # 与模型中相同的k值
                    
                    # PDG稀疏门控
                    pdg_top_k_logits, pdg_indices = torch.topk(pdg_gate_logits, k=k, dim=-1)
                    pdg_top_k_probs = F.softmax(pdg_top_k_logits, dim=-1)
                    pdg_gate_probs = torch.zeros_like(pdg_gate_logits).scatter_(-1, pdg_indices, pdg_top_k_probs)
                    
                    # CFG稀疏门控
                    cfg_top_k_logits, cfg_indices = torch.topk(cfg_gate_logits, k=k, dim=-1)
                    cfg_top_k_probs = F.softmax(cfg_top_k_logits, dim=-1)
                    cfg_gate_probs = torch.zeros_like(cfg_gate_logits).scatter_(-1, cfg_indices, cfg_top_k_probs)
                    
                    # AST稀疏门控
                    ast_top_k_logits, ast_indices = torch.topk(ast_gate_logits, k=k, dim=-1)
                    ast_top_k_probs = F.softmax(ast_top_k_logits, dim=-1)
                    ast_gate_probs = torch.zeros_like(ast_gate_logits).scatter_(-1, ast_indices, ast_top_k_probs)
                else:
                    # 标准softmax门控
                    pdg_gate_probs = F.softmax(pdg_gate_logits, dim=-1)
                    cfg_gate_probs = F.softmax(cfg_gate_logits, dim=-1)
                    ast_gate_probs = F.softmax(ast_gate_logits, dim=-1)
                
                pdg_expert_weights.append(pdg_gate_probs.cpu().numpy())
                cfg_expert_weights.append(cfg_gate_probs.cpu().numpy())
                ast_expert_weights.append(ast_gate_probs.cpu().numpy())
            else:
                # 原始GraphMixtureOfExperts模型
                graph_features_list = [pdg_features, cpg_features, ast_features]
                _, expert_weights = model.graph_moe(graph_features_list)
                
                # 分割专家权重
                num_experts = model.graph_moe.num_experts
                pdg_weights = expert_weights[:, :num_experts].cpu().numpy()
                cfg_weights = expert_weights[:, num_experts:2*num_experts].cpu().numpy()
                ast_weights = expert_weights[:, 2*num_experts:].cpu().numpy()
                
                pdg_expert_weights.append(pdg_weights)
                cfg_expert_weights.append(cfg_weights)
                ast_expert_weights.append(ast_weights)
    
    # 合并所有批次的权重
    pdg_expert_weights = np.vstack(pdg_expert_weights)
    cfg_expert_weights = np.vstack(cfg_expert_weights)
    ast_expert_weights = np.vstack(ast_expert_weights)
    
    # 计算每个专家的平均激活值
    pdg_avg_weights = np.mean(pdg_expert_weights, axis=0)
    cfg_avg_weights = np.mean(cfg_expert_weights, axis=0)
    ast_avg_weights = np.mean(ast_expert_weights, axis=0)
    
    # 计算每个专家的标准差
    pdg_std_weights = np.std(pdg_expert_weights, axis=0)
    cfg_std_weights = np.std(cfg_expert_weights, axis=0)
    ast_std_weights = np.std(ast_expert_weights, axis=0)
    
    # 计算每个样本的最活跃专家
    pdg_top_experts = np.argmax(pdg_expert_weights, axis=1)
    cfg_top_experts = np.argmax(cfg_expert_weights, axis=1)
    ast_top_experts = np.argmax(ast_expert_weights, axis=1)
    
    # 统计每个专家成为最活跃专家的次数
    pdg_expert_counts = np.bincount(pdg_top_experts, minlength=len(pdg_avg_weights))
    cfg_expert_counts = np.bincount(cfg_top_experts, minlength=len(cfg_avg_weights))
    ast_expert_counts = np.bincount(ast_top_experts, minlength=len(ast_avg_weights))
    
    # 转换为百分比
    pdg_expert_percent = pdg_expert_counts / np.sum(pdg_expert_counts) * 100
    cfg_expert_percent = cfg_expert_counts / np.sum(cfg_expert_counts) * 100
    ast_expert_percent = ast_expert_counts / np.sum(ast_expert_counts) * 100
    
    # 准备专家类型标签
    if hasattr(model.config, 'ast_expert_types'):
        pdg_expert_types = model.config.pdg_expert_types
        cfg_expert_types = model.config.cfg_expert_types
        ast_expert_types = model.config.ast_expert_types
    else:
        # 默认标签
        pdg_expert_types = [f"PDG experts{i+1}" for i in range(len(pdg_avg_weights))]
        cfg_expert_types = [f"CFG experts{i+1}" for i in range(len(cfg_avg_weights))]
        ast_expert_types = [f"AST experts{i+1}" for i in range(len(ast_avg_weights))]
    
    # 创建结果统计
    experts_stats = {
        'pdg': {
            'avg_weights': pdg_avg_weights,
            'std_weights': pdg_std_weights,
            'expert_counts': pdg_expert_counts,
            'expert_percent': pdg_expert_percent,
            'expert_types': pdg_expert_types
        },
        'cfg': {
            'avg_weights': cfg_avg_weights,
            'std_weights': cfg_std_weights,
            'expert_counts': cfg_expert_counts,
            'expert_percent': cfg_expert_percent,
            'expert_types': cfg_expert_types
        },
        'ast': {
            'avg_weights': ast_avg_weights,
            'std_weights': ast_std_weights,
            'expert_counts': ast_expert_counts,
            'expert_percent': ast_expert_percent,
            'expert_types': ast_expert_types
        }
    }
    
    # 绘制图表
    if save_path:
        # 设置图表样式
        plt.style.use('seaborn-darkgrid')
        
        # 创建一个3x2的图表网格
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 设置标题
        fig.suptitle('SGMoe Experts Analysis', fontsize=16)
        
        # 绘制平均激活值
        graph_types = ['PDG', 'CFG', 'AST']
        avg_weights = [pdg_avg_weights, cfg_avg_weights, ast_avg_weights]
        expert_types = [pdg_expert_types, cfg_expert_types, ast_expert_types]
        
        for i, (graph_type, weights, types) in enumerate(zip(graph_types, avg_weights, expert_types)):
            # 平均激活值图表
            axes[i, 0].bar(range(len(weights)), weights, yerr=experts_stats[graph_type.lower()]['std_weights'], 
                           color=sns.color_palette("husl", len(weights)))
            axes[i, 0].set_title(f'{graph_type} experts average activation value')
            axes[i, 0].set_xlabel('experts type')
            axes[i, 0].set_ylabel('average activation value')
            axes[i, 0].set_xticks(range(len(weights)))
            axes[i, 0].set_xticklabels(types, rotation=45)
            
            # 专家选择百分比图表
            percentages = experts_stats[graph_type.lower()]['expert_percent']
            axes[i, 1].pie(percentages, labels=types, autopct='%1.1f%%', 
                          colors=sns.color_palette("husl", len(percentages)),
                          startangle=90)
            axes[i, 1].set_title(f'{graph_type} experts selection percentage')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(save_path)
        plt.close()
        
        # 创建热力图显示不同图类型专家的协同工作情况
        plt.figure(figsize=(12, 10))
        
        # 计算PDG与CFG、PDG与AST、CFG与AST专家选择的相关性
        def compute_correlation(expert1, expert2):
            """计算两种专家选择的相关性矩阵"""
            correlation = np.zeros((len(expert1[0]), len(expert2[0])))
            for i in range(len(expert1)):
                max_expert1 = np.argmax(expert1[i])
                max_expert2 = np.argmax(expert2[i])
                correlation[max_expert1, max_expert2] += 1
            return correlation / np.sum(correlation)
        
        # 计算PDG与CFG的相关性
        pdg_cfg_corr = compute_correlation(pdg_expert_weights, cfg_expert_weights)
        
        # 绘制热力图
        plt.subplot(1, 1, 1)
        sns.heatmap(pdg_cfg_corr, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=cfg_expert_types, yticklabels=pdg_expert_types)
        plt.title('PDG experts and CFG experts collaborative activation analysis')
        plt.xlabel('CFG experts type')
        plt.ylabel('PDG experts type')
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_correlation.png'))
        plt.close()
    
    return experts_stats
