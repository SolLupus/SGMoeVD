#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_processing import balance_dataset, analyze_dataset, create_mini_dataset
import sys

def process_bigvul_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    处理BigVul数据集的Parquet格式文件，分割为训练、验证和测试集
    
    参数:
    - input_dir: BigVul数据集目录，包含parquet文件
    - output_dir: 输出CSV文件的目录路径
    - train_ratio, val_ratio, test_ratio: 数据集分割比例
    - seed: 随机种子
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"正在处理BigVul数据集: {input_dir}")
    
    # 查找所有的parquet文件
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        print(f"错误: 未找到parquet文件，请确保数据集目录包含parquet格式的数据文件")
        return None
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 读取所有parquet文件并合并
    all_data = []
    for file_path in tqdm(parquet_files, desc="读取parquet文件"):
        try:
            # 使用pandas读取parquet文件
            df = pd.read_parquet(file_path)
            print(f"成功读取文件: {file_path}, 包含 {len(df)} 行")
            if len(df) > 0:
                # 打印前几行数据示例
                print(f"数据示例:\n{df.head(2)}")
                print(f"列名: {df.columns.tolist()}")
                all_data.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {str(e)}")
    
    if not all_data:
        print("错误: 未能成功读取任何数据")
        return None
    
    # 合并所有数据
    df = pd.concat(all_data, ignore_index=True)
    print(f"合并后的数据共有 {len(df)} 行")
    
    # 检查必要的字段是否存在
    # 根据BigVul数据集文档调整所需列名
    required_columns = ['func_before', 'vul']  # 根据实际情况调整
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    # 如果缺少必要的列，尝试映射存在的列
    if missing_columns:
        print(f"警告: 缺少必要的列: {missing_columns}")
        # 尝试映射常见的BigVul数据集列名
        column_mappings = {
            'func_before': ['func', 'function', 'code', 'source_code', 'before', 'content', 'func_after'],
            'target': ['label', 'is_vulnerable', 'vulnerable', 'buggy', 'is_buggy', 'bug', 'vulnerability']
        }
        
        # 尝试替换缺失的列
        for missing_col in missing_columns:
            for alt_col in column_mappings[missing_col]:
                if alt_col in df.columns:
                    print(f"使用列 {alt_col} 替代 {missing_col}")
                    df[missing_col] = df[alt_col]
                    break
    
    # 再次检查必要的列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: 仍然缺少必要的列: {missing_columns}")
        print(f"可用的列: {df.columns.tolist()}")
        # 尝试使用可用的代码和标签列
        if 'func_before' in missing_columns and 'func_after' in df.columns:
            print("使用 func_after 作为代码列")
            df['func_before'] = df['func_after']
        elif 'func_before' in missing_columns:
            # 寻找其他可能的代码列
            possible_code_cols = [col for col in df.columns if 'func' in col.lower() or 'code' in col.lower()]
            if possible_code_cols:
                first_code_col = possible_code_cols[0]
                print(f"使用 {first_code_col} 作为代码列")
                df['func_before'] = df[first_code_col]
        
        if 'target' in missing_columns:
            # 寻找其他可能的标签列
            possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'bug' in col.lower() or 'vuln' in col.lower()]
            if possible_label_cols:
                first_label_col = possible_label_cols[0]
                print(f"使用 {first_label_col} 作为标签列")
                df['target'] = df[first_label_col]
    
    # 最终检查必要的列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: 无法找到必要的数据列: {missing_columns}，无法继续处理")
        return None
    
    # 数据预处理和清洗
    print("正在预处理数据...")
    # 删除代码字段为空的行
    df = df[df['func_before'].notna()]
    # 确保目标字段为整数类型
    df['vul'] = df['vul'].astype(int)
    
    # 重命名列以符合我们的标准格式
    df = df.rename(columns={'func_before': 'code', 'vul': 'label'})
    
    # 检查类别分布
    labels = df['label'].values
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    print(f"漏洞样本数: {positive_count}, 占比: {positive_count / len(labels):.2%}")
    print(f"非漏洞样本数: {negative_count}, 占比: {negative_count / len(labels):.2%}")
    
    # 分割数据集
    print("正在分割数据集...")
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=seed, stratify=df['label'])
    train, val = train_test_split(
        train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=seed,
        stratify=train_val['label']
    )
    
    # 保存为CSV文件
    print("正在保存数据集...")
    train[['code', 'label']].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val[['code', 'label']].to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test[['code', 'label']].to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # 打印数据集分割信息
    print(f"训练集大小: {len(train)}, 验证集大小: {len(val)}, 测试集大小: {len(test)}")
    print(f"数据集已保存至: {output_dir}")
    
    return {
        'train': train,
        'valid': val,
        'test': test
    }

def main():
    parser = argparse.ArgumentParser(description='处理BigVul漏洞数据集')
    parser.add_argument('--input_dir', type=str, required=True, help='BigVul数据集目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--balance', action='store_true', help='是否平衡数据集')
    parser.add_argument('--balance_method', type=str, choices=['undersample', 'oversample'],
                        default='undersample', help='平衡方法: undersample或oversample')
    parser.add_argument('--create_mini', action='store_true', help='是否创建小型数据集用于测试')
    parser.add_argument('--analyze', action='store_true', help='是否分析数据集统计信息')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理并分割数据集
    try:
        dataset = process_bigvul_dataset(args.input_dir, args.output_dir)
        if dataset is None:
            print("处理数据集失败，请检查数据集格式")
            return
    except Exception as e:
        print(f"处理数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 分析数据集
    if args.analyze and os.path.exists(os.path.join(args.output_dir, 'train.csv')):
        try:
            analyze_dataset(args.output_dir)
        except Exception as e:
            print(f"分析数据集时出错: {str(e)}")
    
    # 平衡数据集
    if args.balance and os.path.exists(os.path.join(args.output_dir, 'train.csv')):
        try:
            balance_dir = os.path.join(args.output_dir, 'balanced')
            balance_dataset(args.output_dir, balance_dir, method=args.balance_method)
            if args.analyze:
                try:
                    analyze_dataset(balance_dir)
                except Exception as e:
                    print(f"分析平衡数据集时出错: {str(e)}")
        except Exception as e:
            print(f"平衡数据集时出错: {str(e)}")
    
    # 创建小型数据集用于测试
    if args.create_mini and os.path.exists(os.path.join(args.output_dir, 'train.csv')):
        try:
            mini_dir = os.path.join(args.output_dir, 'mini')
            create_mini_dataset(args.output_dir, mini_dir)
            if args.analyze:
                try:
                    analyze_dataset(mini_dir)
                except Exception as e:
                    print(f"分析小型数据集时出错: {str(e)}")
        except Exception as e:
            print(f"创建小型数据集时出错: {str(e)}")
    
    print("数据处理完成!")

if __name__ == "__main__":
    main() 