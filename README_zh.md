# 🏗️ 机器学习 / 深度学习 作品集

<p align="center">
  <b>Larry</b> ·
  香港城市大学 ·
  Surveying (BSc) ·
  GitHub
</p>

---

Hi，我是 Larry。我是香港城市大学 Surveying 专业本科生，目前在自学机器学习与深度学习。我的项目围绕数据驱动方法在建造安全、城市监测、基础设施管理等场景的应用展开——这也是我计划在硕士阶段（数字建造 / 智慧城市 / 地理空间数据科学方向）继续深入的方向。

> **声明：** 本项目为个人学习与作品集展示用途。代码实现过程中参考了公开教程、官方文档、社区博客及ai工具。所有使用数据集均通过开源途径获取。项目代码仅供学习交流。

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/Larry582-ttk/my_project
cd my_project

# 安装依赖
pip install -r requirements.txt
```

| 项目 | 运行命令 |
| :--- | :--- |
| 建筑工地事故风险分类 | `cd hk_construction_risk && python logistic_model.py` |
| 摩托车头盔检测 | `cd helmet_detection && python main.py` |
| LSTM 交通流预测 | `cd lstm && python main.py` |

---

## 项目列表

| # | 项目 | 领域 | 核心技术 |
| :-: | :--- | :--- | :--- |
| 1 | 建筑工地事故风险分类 | NLP + 机器学习 | Sentence-BERT, 层次聚类, 逻辑回归, 决策树 |
| 2 | 摩托车头盔检测 | 计算机视觉 | VGG16, ONNX 导出 |
| 3 | LSTM 交通流预测 | 时间序列 | LSTM, 滑动窗口 |

---

## 项目一：建筑工地事故风险分类

📁 `hk_construction_risk/`

将香港劳工处工作安全警示事故标题自动分类到 9 个风险类别。采用无监督聚类 + 有监督分类两阶段策略：先用聚类辅助快速标注，再训练分类器用于新数据预测。

### 数据

| 项目 | 说明 |
| :--- | :--- |
| **来源** | 香港劳工处网站 — 工作安全警示新闻（Work Safety Alerts） |
| **时间范围** | 2012–2024 年 |
| **规模** | 398 条事故标题 |
| **采集方式** | Web 爬取后人工整理 |

### 9 类分类体系

| 代码 | 类别 |
| :--: | :--- |
| 1 | 高处坠落（Fall from Height） |
| 2 | 被坠落物体击中（Struck by Falling Object） |
| 3 | 被车辆/移动物体撞击（Struck by Vehicle / Moving Object） |
| 4 | 被夹/压/困/坍塌（Trapped / Crushed / Caught / Collapse） |
| 5 | 触电（Electrocution） |
| 6 | 有害物质/窒息/中毒（Exposure / Asphyxiation / Poisoning） |
| 7 | 火灾/爆炸/烧伤（Fire / Explosion / Burn） |
| 8 | 溺水（Drowning） |
| 0 | 其他（Others） |

### 流程

```text
爬虫采集 → Sentence-BERT 向量化 → 层次聚类
→ 人工标注 189 条 → 训练分类器（逻辑回归 / 决策树）→ 预测测试集新数据
```

### 文件结构

```text
hk_construction_risk/
├── logistic_model.py              # 逻辑回归模型
├── decisiontree_model.py          # 决策树模型
├── web_scraper.py                 # 爬虫
├── pca_exploration.py             # PCA 降维探索（可选运行）
├── data/
│   ├── raw/                       # 原始数据（含 training / test 集）
│   ├── intermediate/              # 待标注 / 已标注数据
│   └── output/                    # 分类结果
└── models/                        # 保存的模型
```

### 技术栈

`requests`, `BeautifulSoup`, `sentence-transformers`, `scikit-learn`, `joblib`

### 结果（398 条标注后数据）

| 类别 | 数量 | 占比 |
| :--- | :-: | :-: |
| 高处坠落（Fall from Height） | 170 | 42.7% |
| 被夹/压/困/坍塌（Trapped / Crushed / Caught / Collapse） | 91 | 22.9% |
| 被车辆/移动物体撞击（Struck by Vehicle / Moving Object） | 44 | 11.1% |
| 被坠落物体击中（Struck by Falling Object） | 33 | 8.3% |
| 触电（Electrocution） | 21 | 5.3% |
| 有害物质/窒息/中毒（Exposure / Asphyxiation / Poisoning） | 15 | 3.8% |
| 其他（Others） | 11 | 2.8% |
| 溺水（Drowning） | 8 | 2.0% |
| 火灾/爆炸/烧伤（Fire / Explosion / Burn） | 5 | 1.3% |

> **高处坠落（42.7%）和坍塌/夹压（22.9%）合计占事故总数 65% 以上**，是香港建筑工地最主要的致死事故类型。

---

## 项目二：摩托车头盔检测

📁 `helmet_detection/`

简单目标检测任务，基于 VGG16 同时预测图片中单个物体的位置和类别。每张图片仅取第一个标注物体进行训练，输出边界框坐标（x, y, w, h）和 4 个类别概率。

### 数据

| 项目 | 说明 |
| :--- | :--- |
| **任务类型** | 简单目标检测（位置回归 + 4 类别分类） |
| **4 个类别** | `no_helmet`, `motor`, `number`, `with_helmet` |
| **数据格式** | YOLO 格式（txt）（VOC 格式加载代码已实现，当前训练脚本使用 YOLO 格式） |
| **数据规模** | 训练集 144 张，验证集 18 张 |

### 方法

采用 VGG16 作为骨干网络，替换最后一层为 8 个输出的全连接层（前 4 个为边界框坐标，后 4 个为类别概率），使用 SGD 优化器训练。损失函数由位置损失（MSE）和分类损失（CrossEntropy）两部分组成。训练完成后导出为 ONNX 格式。

### 流程

```text
YOLO 数据加载 → 图像预处理 → VGG16 特征提取
→ 8 输出全连接层（4 坐标 + 4 类别）→ SGD 训练 → ONNX 导出
```

### 文件结构

```text
helmet_detection/
├── main.py                        # 主训练脚本
├── model.py                       # VGG16 + 全连接层（输出为 4 坐标 + 4 类别）
├── loss_fn.py                     # 损失函数（MSE + CrossEntropy）
├── utils/
│   ├── load_yolo_dataset.py       # YOLO 格式加载
│   └── load_voc_dataset.py        # VOC 格式加载
├── dataset/
│   ├── HelmetDataset-VOC/         # VOC 格式数据集
│   ├── HelmetDataset-YOLO-Train/  # YOLO 训练集
│   └── HelmetDataset-YOLO-Val/    # YOLO 验证集
└── outputs/                       # 模型输出目录
```

### 技术栈

`PyTorch`, `torchvision`, `ONNX`, `NumPy`

**致谢：** 本项目代码实现参考了 [小土堆（xiaotudui）](https://github.com/xiaotudui/tudui-object-detection-model) 的 YOLO 教程（MIT License），在此致谢。

---

## 项目三：LSTM 交通流预测

📁 `lstm/`

基于加州高速公路交通流数据，使用 LSTM 预测节点流量。

### 数据

| 项目 | 说明 |
| :--- | :--- |
| **数据集** | PeMSD4（加州交通局性能测量系统） |
| **节点数量** | 307 个传感器节点 |
| **原始特征** | flow（流量）、occupy（占有率）、speed（速度）—— 仅使用 flow 进行单变量预测 |
| **时间粒度** | 5 分钟 / 采样点 |
| **选取节点** | node_id = 1 |
| **样本构造** | 滑动窗口（history = 12 → future = 1），使用 StandardScaler 对特征和标签分别标准化 |

### 方法

双层单向LSTM（隐藏层维度 64），按时间顺序划分训练集（前 80%）和测试集（后 20%）。训练 100 个 epoch，batch_size = 128，学习率 0.001，使用 Adam 优化器，MSELoss 作为损失函数。

### 结果

| 指标 | 数值 |
| :--- | :-: |
| R² | 0.9167 |
| MAE | 24.12 |

> 模型解释了 **91.67%** 的数据方差。

### 流程

```text
原始数据（仅取flow做预测）→ StandardScaler 标准化 → 滑动窗口构造样本 (12→1)
→ 按时间顺序划分 (8:2) → LSTM 训练 → R² / MAE 评估 → 结果可视化
```

### 文件结构

```text
lstm/
├── main.py                        # 主训练脚本
├── model.py                       # LSTM 模型
├── loss_fn.py                     # 损失函数
├── utils/data_utils.py            # 数据加载与预处理（标准化、滑动窗口构造样本）
├── dataset/pems04.npz             # 交通流数据集
└── outputs/                       # 结果图（预测对比图 + 散点图）
```

### 技术栈

`PyTorch`, `NumPy`, `pandas`, `scikit-learn`, `Matplotlib`

---

## 技术栈总览

| 方向 | 工具 |
| :--- | :--- |
| 机器学习 / 深度学习 | PyTorch, scikit-learn, sentence-transformers |
| 数据处理与可视化 | pandas, NumPy, Matplotlib |
| 其他工具 | Git, ONNX, joblib, BeautifulSoup |

---

<p align="center">
  © 2026 Larry · 作品集
</p>
