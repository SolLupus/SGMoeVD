# 代码漏洞检测系统

本系统是一个基于多图神经网络的代码漏洞检测工具，结合了程序依赖图(PDG)、控制流图(CFG)和抽象语法树(AST)来进行分析，支持GCN和GAT模型架构。

## 功能特点

- **多图分析**: 使用PDG、CFG和AST三种不同的代码表示进行综合分析
- **图神经网络**: 采用GCN和GAT模型对图结构进行编码
- **专业化混合专家系统**: 为不同类型的图设计不同的专家模型
- **跨图注意力机制**: 捕获不同类型图之间的交互关系
- **可解释性**: 通过专家模型权重分布展示模型决策过程
- **模型混合**: 采用稀疏门控混合专家系统(MoE)实现多图特征的有效融合

## 系统架构

系统主要由以下几个部分组成：

1. **数据预处理模块**: 将源代码解析为PDG、CFG和AST图结构
2. **图神经网络编码器**: 对图结构进行特征提取，支持GCN和GAT两种架构
3. **混合专家模块**: 融合多图特征，实现高效的漏洞检测
4. **Web前端界面**: 提供用户友好的交互界面，展示检测结果和图可视化

## 安装与运行

### 依赖安装

```bash
pip install -r requirements.txt
```

### 运行前端应用

```bash
python app.py
```

系统默认在 http://localhost:5000 启动，可以通过浏览器访问。

## 使用方法

1. 在代码输入框中输入或粘贴C/C++代码，或点击"加载示例代码"按钮
2. 点击"分析代码"按钮启动检测
3. 系统将显示检测结果，包括：
   - 是否存在漏洞及置信度
   - 如有漏洞，显示漏洞类型和风险等级
   - 代码的PDG、CFG和AST图结构可视化
   - 专家模型的权重分布，展示决策过程

## 项目结构

```
.
├── app.py                    # Web应用入口
├── configs.py                # 配置文件
├── data_processing.py        # 数据处理模块
├── data_processing_utils.py  # 数据处理辅助函数
├── main.py                   # 项目主入口
├── models.py                 # 模型定义
├── train.py                  # 训练脚本
├── utils.py                  # 工具函数
├── process_bigvul.py         # BigVul数据集处理
├── process_chromedebain.py   # Chrome/Debian数据集处理
├── requirements.txt          # 依赖文件
├── static/                   # 静态资源
│   ├── css/                  # CSS样式
│   └── js/                   # JavaScript文件
└── templates/                # HTML模板
    └── index.html            # 前端主页
```

## 注意事项

- 系统需要安装Python 3.7+
- 需要安装PyTorch和PyTorch Geometric
- 对于生产环境，建议使用GPU进行加速

## 模型架构

该模型融合了三种代码表示图：
- PDG (Program Dependence Graph) - 程序依赖图
- CPG (Control Flow Graph) - 控制流图
- AST (Abstract Syntax Tree) - 抽象语法树

### GCN与GAT的区别

- **GCN (图卷积网络)**: 将卷积神经网络应用于图结构数据，对所有邻居节点赋予相同的权重。
- **GAT (图注意力网络)**: 引入注意力机制，自适应地为不同邻居节点分配不同的权重，能够更好地处理复杂的图结构。

## 使用方法

### 训练模型

```bash
# 使用GAT模型（默认）
python main.py

# 使用GCN模型
python main.py --model gcn

# 使用GAT模型并指定注意力头数
python main.py --model gat --heads 8

# 指定批处理大小
python main.py --batch_size 128
```

### 命令行参数

- `--model`: 选择使用的模型架构，可选值为'gat'或'gcn'，默认为'gat'
- `--heads`: GAT模型的注意力头数，默认为4
- `--batch_size`: 训练批处理大小，默认为256

## 性能比较

与GCN相比，GAT模型通常具有以下优势：
1. 通过注意力机制学习节点间的重要性权重
2. 更好地捕捉图结构中的重要模式
3. 对噪声和不相关连接有更强的鲁棒性

在漏洞检测任务中，GAT能够更精确地关注代码中可能存在漏洞的关键部分，提高检测准确率。 