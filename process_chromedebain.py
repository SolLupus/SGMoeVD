#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_processing import process_json_dataset, balance_dataset, analyze_dataset, create_mini_dataset
import sys

def convert_chromedebain_format(input_dir, output_file):
    """
    将ChromeDebain目录下的数据转换为项目需要的JSON格式
    
    参数:
        input_dir: ChromeDebain数据集目录
        output_file: 输出的JSON文件路径
    """
    print(f"正在处理ChromeDebain数据集: {input_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在!")
        return []
    
    # 查找vulnerable.json和non_vulnerable.json文件
    vulnerable_path = os.path.join(input_dir, "vulnerables.json")
    non_vulnerable_path = os.path.join(input_dir, "non-vulnerables.json")
    
    if not os.path.exists(vulnerable_path):
        print(f"错误: 未找到漏洞数据文件 {vulnerable_path}")
        vulnerable_samples = []
    else:
        print(f"找到漏洞数据文件: {vulnerable_path}")
    
    if not os.path.exists(non_vulnerable_path):
        print(f"错误: 未找到非漏洞数据文件 {non_vulnerable_path}")
        non_vulnerable_samples = []
    else:
        print(f"找到非漏洞数据文件: {non_vulnerable_path}")
    
    # 存储所有样本
    all_samples = []
    
    # 处理漏洞数据
    if os.path.exists(vulnerable_path):
        try:
            with open(vulnerable_path, 'r', encoding='utf-8') as f:
                vulnerable_data = json.load(f)
                
            print(f"成功读取漏洞数据文件，包含 {len(vulnerable_data)} 个样本")
            
            # 处理每个漏洞样本，标签设为1
            for i, code in enumerate(vulnerable_data):
                if isinstance(code, str) and len(code.strip()) > 0:
                    sample = {
                        'func': code,
                        'target': 1,  # 漏洞标签设为1
                        'project': 'chromedebain',
                        'commit_id': f'vulnerable_{i}'
                    }
                    all_samples.append(sample)
                elif isinstance(code, dict) and 'code' in code:
                    # 处理已经是字典格式的情况
                    sample = {
                        'func': code['code'],
                        'target': 1,  # 漏洞标签设为1
                        'project': code.get('project', 'chromedebain'),
                        'commit_id': code.get('commit_id', f'vulnerable_{i}')
                    }
                    all_samples.append(sample)
            
            print(f"处理了 {len(all_samples)} 个漏洞样本")
            
        except Exception as e:
            print(f"处理漏洞数据文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 处理非漏洞数据
    if os.path.exists(non_vulnerable_path):
        try:
            with open(non_vulnerable_path, 'r', encoding='utf-8') as f:
                non_vulnerable_data = json.load(f)
                
            print(f"成功读取非漏洞数据文件，包含 {len(non_vulnerable_data)} 个样本")
            
            # 处理每个非漏洞样本，标签设为0
            non_vulnerable_count = 0
            for i, code in enumerate(non_vulnerable_data):
                if isinstance(code, str) and len(code.strip()) > 0:
                    sample = {
                        'func': code,
                        'target': 0,  # 非漏洞标签设为0
                        'project': 'chromedebain',
                        'commit_id': f'non_vulnerable_{i}'
                    }
                    all_samples.append(sample)
                    non_vulnerable_count += 1
                elif isinstance(code, dict) and 'code' in code:
                    # 处理已经是字典格式的情况
                    sample = {
                        'func': code['code'],
                        'target': 0,  # 非漏洞标签设为0
                        'project': code.get('project', 'chromedebain'),
                        'commit_id': code.get('commit_id', f'non_vulnerable_{i}')
                    }
                    all_samples.append(sample)
                    non_vulnerable_count += 1
            
            print(f"处理了 {non_vulnerable_count} 个非漏洞样本")
            
        except Exception as e:
            print(f"处理非漏洞数据文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 检查处理结果
    print(f"总共收集了 {len(all_samples)} 个有效样本")
    
    if len(all_samples) == 0:
        print("警告: 未找到任何有效样本! 请检查数据集格式是否正确。")
        return all_samples
    
    # 保存为统一的JSON格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存至: {output_file}")
    return all_samples

def custom_process_json_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    处理JSON格式的漏洞检测数据集，分割为训练、验证和测试集
    处理空数据集的情况

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
    
    # 处理空数据集的情况
    if len(samples) == 0:
        print("警告: 处理后的样本数为0，创建空的CSV文件...")
        empty_df = pd.DataFrame(columns=['code', 'label'])
        empty_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        empty_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
        empty_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        return {
            'train': empty_df,
            'valid': empty_df,
            'test': empty_df
        }

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

def main():
    parser = argparse.ArgumentParser(description='处理ChromeDebain漏洞数据集')
    parser.add_argument('--input_dir', type=str, required=True, help='ChromeDebain数据集目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--temp_json', type=str, default='chromedebain_processed.json', help='中间JSON文件名')
    parser.add_argument('--balance', action='store_true', help='是否平衡数据集')
    parser.add_argument('--balance_method', type=str, choices=['undersample', 'oversample'],
                        default='undersample', help='平衡方法: undersample或oversample')
    parser.add_argument('--create_mini', action='store_true', help='是否创建小型数据集用于测试')
    parser.add_argument('--analyze', action='store_true', help='是否分析数据集统计信息')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换数据集格式
    temp_json_path = os.path.join(args.output_dir, args.temp_json)
    samples = convert_chromedebain_format(args.input_dir, temp_json_path)
    
    if len(samples) == 0:
        print("警告: 未找到任何有效样本，生成空的数据集...")
        
    # 处理并分割数据集
    try:
        dataset = custom_process_json_dataset(temp_json_path, args.output_dir)
    except Exception as e:
        print(f"处理JSON数据集时出错: {str(e)}")
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