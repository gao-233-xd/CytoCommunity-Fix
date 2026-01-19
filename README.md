# CytoCommunity-Supervised-Sparse (显存优化版)

本项目是 **CytoCommunity 监督学习 (Supervised) 版本** 的修改版。

本次修改的核心目的是解决在处理大规模空间组学数据时遇到的 **显存溢出 (Out-Of-Memory, OOM)** 问题。我们通过将底层的图数据结构从 **Dense (稠密图)** 格式迁移至 **Sparse (稀疏图)** 格式，并使用 `sparse_mincut_pool` 替换了原有的 `dense_mincut_pool`，从而显著降低了计算资源消耗。

## 🛠 主要修改内容

修改主要集中在 **Step 1** 和 **Step 2**，旨在实现稀疏图计算，同时保持与后续分析步骤（Step 3 & Step 4）的完全兼容。

### 1. Step 1: 图构建 (`Step1_ConstructCellularSpatialGraphs.py`)
* **移除稠密化转换：**
    * **原版：** 使用了 `transform=T.ToDense(max_nodes)`，这会强制将所有样本图填充（Padding）至最大节点数，导致巨大的内存浪费。
    * **修改后：** 移除了 `ToDense` 转换。数据集现在以默认的 **Sparse (Edge Index)** 格式加载，大幅减少了内存占用。

### 2. Step 2: TCN 模型训练 (`Step2_TCNLearning_Supervised.py`)
* **模型架构升级 (Dense $\to$ Sparse)：**
    * 将 `DenseGraphConv` 替换为标准的 **`GraphConv`**。
    * 将 `DenseDataLoader` 替换为标准的 **`DataLoader`**。
* **池化机制更新：**
    * 使用 **`sparse_mincut_pool`** 替换了原版的 `dense_mincut_pool`。
    * 重写了前向传播（Forward）逻辑，以支持稀疏 Batch 的处理，并手动重构了池化后的边索引（Edge Index）。
* **输出兼容性适配：**
    * 尽管内部计算转为稀疏模式，但我们在输出阶段对文件（如 `ClusterAssignMatrix`, `NodeMask`）进行了格式适配，确保能 **无缝衔接** 原流程的 **Step 3 (R语言脚本)** 和 **Step 4 (可视化)**，无需修改后续代码。

