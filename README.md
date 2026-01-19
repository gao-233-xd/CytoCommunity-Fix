# CytoCommunity-Sparse (显存优化版)

本项目是 **CytoCommunity** 的深度优化版本，包含 **监督学习 (Supervised)** 和 **无监督学习 (Unsupervised)** 两个模块。

## 💡 项目背景
原版 CytoCommunity 在处理大规模空间组学数据时，由于采用 **Dense (稠密图)** 格式存储邻接矩阵（Adjacency Matrix），容易导致 **显存溢出 (Out-Of-Memory, OOM)**。

本项目将底层的图数据结构全面迁移至 **Sparse (稀疏图)** 格式，并针对稀疏训练的特性对模型架构和训练策略进行了适配。这使得模型能够在有限的显存下处理更大规模的图像数据。

## 🛠 核心修改内容

### 1. 公共部分：图构建 (`Step1`)
* **文件：** `Step1_ConstructCellularSpatialGraphs.py`
* **修改点：** 移除了 `transform=T.ToDense(max_nodes)` 转换。
* **效果：** 不再强制将所有样本图填充（Padding）至最大节点数，而是直接以 PyG 默认的 **Sparse (Edge Index)** 格式加载数据，大幅减少内存占用。

### 2. 监督学习模块 (`Step2 Supervised`)
* **文件：** `Step2_TCNLearning_Supervised.py`
* **架构升级：**
    * `DenseGraphConv` $\to$ **`GraphConv`** (标准稀疏卷积)。
    * `DenseDataLoader` $\to$ **`DataLoader`**。
* **池化层：** 使用 **`sparse_mincut_pool`** 替换了原版的 `dense_mincut_pool`，并重写了前向传播逻辑以支持稀疏 Batch。
* **输出适配：** 尽管内部计算为稀疏模式，输出文件（如 `NodeMask`）已做格式适配，确保与 Step 3 的 R 语言脚本完全兼容。

### 3. 无监督学习模块 (`Step2 Unsupervised`)
* **文件：** `Step2_TCNLearning_Unsupervised.py`
* **架构升级：**
    * `DenseGraphConv` $\to$ **`GCNConv`**。
    * 引入 `sparse_mincut_pool_batch` 处理无监督池化任务。
* **训练策略优化 (重要)：**
    * **学习率 (LR)：** 调整为 **`0.0005`** (原版 0.0001)，以改善稀疏模型在无监督任务下的收敛效果。
    * **保存逻辑：** 修改了原版严格的 Loss 截断机制（原版要求 Loss < -0.6），改为 **发散检测机制**（仅在 Loss 异常过大时重试），提高了训练稳定性，防止死循环。

