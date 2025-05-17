# main.py
import torch.multiprocessing as mp
import os
import sys
# 现在导入其余模块
from configs import Config
from train import train
# from bigvul_train import train_bigvul
# 设置多进程启动方法为'spawn'以支持CUDA
if __name__ == '__main__':
    # 在任何其他导入或代码执行前设置
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # 方法已设置，忽略错误
        pass
    

    
    # 设置环境变量，禁用tokenizer并行
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 设置CUDA可见设备
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 执行训练
    config = Config()
    model, test_metrics = train(config)

    print("\nTraining completed!")
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
