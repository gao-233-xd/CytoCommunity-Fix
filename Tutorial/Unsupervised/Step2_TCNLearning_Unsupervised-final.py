import torch
import torch.nn.functional as F
from torch.nn import Linear
# from torch_geometric.loader import DenseDataLoader
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.nn import GCNConv
# 确保你目录下有 sparse_mincut_pool.py 这个文件，否则这里会报错
from sparse_mincut_pool import sparse_mincut_pool_batch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import os
import numpy as np
import pandas as pd  # 这就是你刚才报错缺少的库
import datetime
import csv
import shutil

## ================= Hyperparameters =================
InputFolderName = "./MERFISH-Brain_Input/"
Image_Name = "1_-0.14"
Num_TCN = 9
Num_Run = 20 
Num_Epoch = 3000 
Embedding_Dimension = 128
Learning_Rate = 0.0001
Loss_Cutoff = -0.6   # 质量控制阈值
Max_Retries = 3      # 最大重试次数

## ================= Data Loading =================
## Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
        Region_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["Image"],  # set our own names for the columns
    )

## Load dataset from the constructed Dataset.
LastStep_OutputFolderName = "./Step1_Output/"

class SpatialOmicsImageDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']

    def download(self):
        pass
    
    def process(self):
        # Read data_list into huge `Data` list.
        # 注意：这里假设 dataset 已经在 Step1 处理好了，直接读取即可
        pass 

# 保持稀疏读取
dataset = SpatialOmicsImageDataset(LastStep_OutputFolderName)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
        super(Net, self).__init__()
        # 使用 GCNConv (稀疏) 
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False, add_self_loops=False)
        num_cluster1 = Num_TCN
        self.pool1 = Linear(hidden_channels, num_cluster1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        s = self.pool1(x) 
        # 使用稀疏池化
        x_pool, adj_pool, mc1, o1 = sparse_mincut_pool_batch(x, edge_index, s, batch)
        
        ClusterAssignTensor_1 = s
        ClusterAdjTensor_1 = adj_pool

        return F.log_softmax(x_pool, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1


def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # 传入 x, edge_index, batch
        out, mc_loss, o_loss, _, _ = model(data.x, data.edge_index, data.batch)
        loss = mc_loss + o_loss
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all


# Extract a single graph for TCN learning.
ThisStep_OutputFolderName = "./Step2_Output_" + Image_Name + "/"
os.makedirs(ThisStep_OutputFolderName, exist_ok=True)

try:
    train_index = [region_name_list["Image"].values.tolist().index(Image_Name)]
except ValueError:
    print(f"Error: Image '{Image_Name}' not found in {Region_filename}")
    exit()

train_dataset = dataset[train_index]
train_loader = DataLoader(train_dataset, batch_size=1)
all_sample_loader = DataLoader(train_dataset, batch_size=1)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

run_number = 1
current_retry = 0 # 初始化重试计数器

while run_number <= Num_Run:

    print(f"This is Run{run_number:02d} (Retry: {current_retry}/{Max_Retries})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
    
    RunFolderName = ThisStep_OutputFolderName + "Run" + str(run_number)
    if os.path.exists(RunFolderName):
        shutil.rmtree(RunFolderName)
    os.makedirs(RunFolderName)
    
    filename_0 = RunFolderName + "/Epoch_UnsupervisedLoss.csv"
    headers_0 = ["Epoch", "UnsupervisedLoss"]
    with open(filename_0, "w", newline='') as f0:
        f0_csv = csv.writer(f0)
        f0_csv.writerow(headers_0)

    previous_loss = float("inf")
    is_diverged = False # 标记是否梯度爆炸

    for epoch in range(1, Num_Epoch+1):
        train_loss = train(epoch)

        if epoch % 100 == 0 or epoch == 1:
             print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")

        with open(filename_0, "a", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow([epoch, train_loss])
        
        # 梯度爆炸检查
        if train_loss > 5.0:
            print(f"  >>> Run {run_number} diverged (Loss > 5.0). Stopping early.")
            is_diverged = True
            break 

        if train_loss == 0 and train_loss == previous_loss:
            break
        else:
            previous_loss = train_loss

    print(f"Final train loss is {train_loss:.4f}")

    # ================= 核心判断逻辑 =================
    
    # 判断是否失败：1. 梯度爆炸 或 2. Loss 没降到 Cutoff 以下
    if is_diverged or (train_loss >= Loss_Cutoff):
        if current_retry < Max_Retries:
            print(f"Run {run_number} Failed (Loss={train_loss:.4f}). Retrying ({current_retry+1}/{Max_Retries})...")
            shutil.rmtree(RunFolderName) # 删除当前失败的结果
            current_retry += 1           # 增加重试计数
            continue                     # 跳过后续保存，重新开始
        else:
            print(f"Run {run_number} Failed multiple times. Max retries reached. Forcing save.")
            # 达到最大重试次数，强制保存结果，不continue
    
    # ================= 保存结果 =================
    for EachData in all_sample_loader:
        EachData = EachData.to(device)
        with torch.no_grad():
            TestModelResult = model(EachData.x, EachData.edge_index, EachData.batch)

        # 1. Assignment Matrix
        ClusterAssignMatrix1 = TestModelResult[3]
        ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)
        ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().detach().numpy() # 加上 detach
        filename1 = RunFolderName + "/TCN_AssignMatrix1.csv"
        np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

        # 2. Adjacency Matrix
        # sparse_mincut_pool_batch 返回的是 [Batch, K, K]，这里取第0个样本
        ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
        ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().detach().numpy()
        filename2 = RunFolderName + "/TCN_AdjMatrix1.csv"
        np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

        # 3. Node Mask
        num_nodes = EachData.x.size(0)
        NodeMask = np.ones((num_nodes,), dtype=int)
        filename3 = RunFolderName + "/NodeMask.csv"
        np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')

    # 成功完成或强制保存后，进入下一个 Run
    run_number = run_number + 1
    current_retry = 0 # 重置计数器

print("All runs finished!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))