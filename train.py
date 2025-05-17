# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
import time
from tqdm import tqdm
import numpy as np
import warnings
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 抑制所有警告
warnings.filterwarnings('ignore')


from configs import Config
from models import EnhancedVulnerabilityDetector
from data_processing import create_dataloaders
from utils import evaluate_model, plot_training_metrics, analyze_moe_experts

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建带有预热的线性学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(config):
    """
    训练模型
    
    参数：
        config: 配置对象
    """
    # 设置随机种子
    torch.manual_seed(4307)
    np.random.seed(42)

    # 确保模型保存目录存在
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 创建数据加载器
    train_loader, valid_loader, test_loader, class_weights = create_dataloaders(config)
    
    # 使用配置中定义的类别权重，而不是数据派生的权重
    if hasattr(config, 'pos_weight') and hasattr(config, 'neg_weight'):
        print(f"使用手动设置的类别权重: 负类={config.neg_weight}, 正类={config.pos_weight}")
        class_weights = torch.tensor([config.neg_weight, config.pos_weight]).to(config.device)
    else:
        # 将数据派生的类别权重移动到设备
        print("使用数据派生的类别权重")
        class_weights = class_weights.to(config.device)

    # 创建增强的模型
    model = EnhancedVulnerabilityDetector(config).to(config.device)

    # 优化器 - 分组参数，对不同部分使用不同的学习率
    # 预训练模型部分使用较小的学习率
    optimizer_grouped_parameters = [
        {
            'params': model.parameters(),
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    # 计算总训练步数
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    # 学习率调度器 - 带预热的线性衰减
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 训练记录
    train_losses = []
    ce_losses = []
    val_metrics_history = []
    train_metrics_history = []
    total = 0
    correct = 0
    # 当前最佳模型指标
    best_f1 = 0.0
    best_accuracy = 0.0
    best_combined_score = 0.0  # 新增：F1和准确率的加权组合分数
    patience = config.patience
    patience_counter = 0
    
    # 训练开始时间
    start_time = time.time()
    global_step = 0

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        epoch_ce_losses = []
        all_epoch_preds = []
        all_epoch_labels = []
        
        # 训练一个epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        
        for batch in progress_bar:
            global_step += 1
            
            # 处理图数据 - 包含pdg, cpg, ast三种图
            graphs = {
                'pdg': batch['graphs']['pdg'].to(config.device),
                'cpg': batch['graphs']['cpg'].to(config.device),
                'ast': batch['graphs']['ast'].to(config.device)
            }
            
            labels = batch['labels'].to(config.device)

            # 前向传播获取logits
            logits, gate_probs, expert_usage_stats = model(graphs)
                
            # 计算分类损失
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            ce_loss = loss
            
            # 计算专家平衡损失 - 使用重要性加权后的交叉熵
            if hasattr(config, 'use_sparse_moe') and config.use_sparse_moe:
                # 对于每种图，计算专家使用的不平衡性
                pdg_usage = expert_usage_stats['pdg']
                cfg_usage = expert_usage_stats['cfg']
                ast_usage = expert_usage_stats['ast']
                
                # 计算每个专家的重要性得分 (越不均衡，得分越高)
                pdg_importance = F.softmax(1.0 / (pdg_usage + 1e-10), dim=0)
                cfg_importance = F.softmax(1.0 / (cfg_usage + 1e-10), dim=0)
                ast_importance = F.softmax(1.0 / (ast_usage + 1e-10), dim=0)
                
                # 计算加权专家使用率
                weighted_pdg_usage = pdg_usage * pdg_importance
                weighted_cfg_usage = cfg_usage * cfg_importance
                weighted_ast_usage = ast_usage * ast_importance
                
                # 计算负载均衡损失
                load_balance_loss = -(torch.sum(weighted_pdg_usage) + 
                                    torch.sum(weighted_cfg_usage) + 
                                    torch.sum(weighted_ast_usage))
                
                # 添加到总损失
                loss = ce_loss + config.load_balance_coef * load_balance_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # 更新学习率
            
            # 记录损失
            epoch_losses.append(loss.item())
            epoch_ce_losses.append(ce_loss.item())
            
            # 收集预测和标签用于计算整体指标
            preds = torch.argmax(logits, dim=1)
            all_epoch_preds.extend(preds.detach().cpu().numpy())
            all_epoch_labels.extend(labels.detach().cpu().numpy())
            
            total += len(labels)
            correct += (preds == labels).sum().item()
            
            # 计算当前整体的准确率、精确率、召回率和F1值
            if len(all_epoch_preds) > 0:
                epoch_accuracy = accuracy_score(all_epoch_labels, all_epoch_preds)
                epoch_precision = precision_score(all_epoch_labels, all_epoch_preds, zero_division=0)
                epoch_recall = recall_score(all_epoch_labels, all_epoch_preds, zero_division=0)
                epoch_f1 = f1_score(all_epoch_labels, all_epoch_preds, zero_division=0)
            else:
                epoch_accuracy = epoch_precision = epoch_recall = epoch_f1 = 0.0
                
            # 更新进度条显示整体训练指标
            progress_bar.set_postfix({
                'loss': np.mean(epoch_losses[-10:]),
                'acc': epoch_accuracy,
                'prec': epoch_precision,
                'rec': epoch_recall,
                'f1': epoch_f1,
                'lr': scheduler.get_last_lr()[0]
            })

        # 计算平均损失
        avg_train_loss = np.mean(epoch_losses)
        avg_ce_loss = np.mean(epoch_ce_losses)
        train_metrics = {
            "accuracy": epoch_accuracy,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "f1": epoch_f1,
        }
        
        train_losses.append(avg_train_loss)
        ce_losses.append(avg_ce_loss)
        
        train_metrics_history.append(train_metrics)
        # 计算当前学习率
        current_lr = scheduler.get_last_lr()[0]

        # 在验证集上评估，每个epoch末尾打印详细信息
        val_metrics = evaluate_model(model, valid_loader, config.device, print_details=True)
        val_metrics_history.append(val_metrics)
        current_f1 = val_metrics['f1']
        current_accuracy = val_metrics['accuracy']

        # 早停检查 - 使用综合评分
        improved = False
        
        if current_f1 > best_f1:
            best_f1 = current_f1  # 记录历史最佳F1
            best_accuracy = current_accuracy  # 记录历史最佳准确率
            improved = True
            
        if improved:
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(config.model_save_path, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'best_accuracy': best_accuracy,
                'best_combined_score': best_combined_score,
            }, best_model_path)
            print(f"保存新的最佳模型，F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"性能未提升，耐心计数: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # 训练时间统计
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        epoch_time = elapsed / (epoch + 1)
        remaining = epoch_time * (config.num_epochs - epoch - 1)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        # 打印进度
        print(f"Epoch {epoch + 1}/{config.num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"CE Loss: {avg_ce_loss:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val Prec: {val_metrics['precision']:.4f}, "
              f"Val Rec: {val_metrics['recall']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, "
              f"LR: {current_lr:.6f}, "
              f"Elapsed: {elapsed_str}, Remaining: {remaining_str}")

    # 绘制训练曲线
    plot_training_metrics(
        train_losses,
        train_metrics_history,
        val_metrics_history,
        os.path.join(config.log_dir, 'training_metrics.png'),
        ce_losses=ce_losses,
    )

    # 加载最佳模型进行测试
    best_model_path = os.path.join(config.model_save_path, 'best_model.pth')
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n===== 最终测试结果 =====")
    print(f"使用来自第 {checkpoint['epoch'] + 1} 轮的最佳模型 (F1: {checkpoint['best_f1']:.4f})")
    test_metrics = evaluate_model(model, test_loader, config.device, print_details=True)
    
    # 如果启用了专业化MoE，分析专家选择情况
    if hasattr(config, 'specialized_moe') and config.specialized_moe:
        print("\n===== 专业化MoE专家分析 =====")
        try:
            experts_analysis_path = os.path.join(config.log_dir, 'moe_experts_analysis.png')
            experts_stats = analyze_moe_experts(
                model, 
                test_loader, 
                config.device, 
                save_path=experts_analysis_path
            )
            print(f"专家分析结果已保存至 {experts_analysis_path}")
            
            # 将专家统计信息添加到测试指标中
            test_metrics['experts_stats'] = {
                k: {
                    'avg_weights': v['avg_weights'].tolist(),
                    'expert_percent': v['expert_percent'].tolist(),
                    'expert_types': v['expert_types']
                } for k, v in experts_stats.items()
            }
        except Exception as e:
            print(f"专家分析出错: {str(e)}")
            print("跳过专家分析，继续执行程序")
            import traceback
            traceback.print_exc()
    
    # 保存测试结果
    results = {
        'test_metrics': test_metrics,
        'best_epoch': checkpoint['epoch'] + 1,
        'best_f1': checkpoint['best_f1'],
        'training_time': elapsed_str,
        'config': {k: str(v) for k, v in config.__dict__.items() if not k.startswith('__')}
    }
    
    with open(os.path.join(config.log_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"测试结果已保存至 {os.path.join(config.log_dir, 'test_results.json')}")
    
    # 保存最终模型（包含完整信息）
    final_save_path = os.path.join(config.model_save_path, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'test_metrics': test_metrics,
        'best_epoch': checkpoint['epoch'] + 1,
        'best_f1': checkpoint['best_f1'],
        'best_loss': checkpoint.get('best_loss', float('inf')),
    }, final_save_path)
    
    print(f"最终模型已保存至 {final_save_path}")

    return model, test_metrics

# if __name__ == "__main__":
#     # 创建配置对象
#     config = Config()
    
#     # 开始训练
#     model, test_metrics = train(config)
