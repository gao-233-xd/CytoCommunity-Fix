import torch
import torch.nn.functional as F
from torch.nn import Linear
#from torch_geometric.loader import DenseDataLoader
from torch_geometric.loader import DataLoader
#from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.nn import GCNConv
from sparse_mincut_pool import sparse_mincut_pool_batch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import os
import numpy as np
import pandas as pd
import datetime
import csv
import shutil


## Hyperparameters
InputFolderName = "./MERFISH-Brain_Input/"
Image_Name = "1_-0.14"
Num_TCN = 9
Num_Run = 20 
Num_Epoch = 3000 
Embedding_Dimension = 128
Learning_Rate = 0.0005 #之前0.0001,但是0.0001会导致结果只有3种细胞
#Loss_Cutoff = -0.6
Max_Retries = 3       # 设置最大重试次数，防止无限死循环

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
#稀疏的情况下不需要max了
# MaxNumNodes_filename = LastStep_OutputFolderName + "MaxNumNodes.txt"
# max_nodes = np.loadtxt(MaxNumNodes_filename, dtype = 'int64', delimiter = "\t").item()

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
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#去掉 transform=T.ToDense(...)，保持稀疏格式
dataset = SpatialOmicsImageDataset(LastStep_OutputFolderName)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
        super(Net, self).__init__()

        #使用 GCNConv (稀疏) 替换 DenseGraphConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_cluster1 = Num_TCN   #This is a hyperparameter.
        self.pool1 = Linear(hidden_channels, num_cluster1)

    def forward(self, x, edge_index, batch):#接受 edge_index 和 batch

        x = F.relu(self.conv1(x, edge_index))
        s = self.pool1(x)  #Here "s" is a non-softmax tensor.
        #使用师兄封装好的
        x_pool, adj_pool, mc1, o1 = sparse_mincut_pool_batch(x, edge_index, s, batch)
        #Save important clustering results_1.
        ClusterAssignTensor_1 = s
        ClusterAdjTensor_1 = adj_pool#池化后的粗粒度图邻接矩阵

        return F.log_softmax(x_pool, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1


def train(epoch):
    model.train()
    loss_all = 0
    mc_all = 0
    o_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #传入 x, edge_index, batch (如果是单图batch就是全0向量)
        out, mc_loss, o_loss, _, _ = model(data.x, data.edge_index, data.batch)
        loss = mc_loss + o_loss#【不对吧，不应该有mc_loss,o_loss同时存在吧】
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all, mc_all, o_all


# Extract a single graph for TCN learning.
ThisStep_OutputFolderName = "./Step2_Output_" + Image_Name + "/"
os.makedirs(ThisStep_OutputFolderName, exist_ok=True)

train_index = [region_name_list["Image"].values.tolist().index(Image_Name)]
train_dataset = dataset[train_index]
#使用普通的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=1)
all_sample_loader = DataLoader(train_dataset, batch_size=1)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
run_number = 1
while run_number <= Num_Run:  #Generate multiple independent runs for ensemble.

    print(f"This is Run{run_number:02d}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, 1).to(device)  #Initializing the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
    
    RunFolderName = ThisStep_OutputFolderName + "Run" + str(run_number)
    if os.path.exists(RunFolderName):
        shutil.rmtree(RunFolderName)
    os.makedirs(RunFolderName)  #Creating the Run folder.
    
    filename_0 = RunFolderName + "/Epoch_UnsupervisedLoss.csv"
    ## 增加列
    headers_0 = ["Epoch", "UnsupervisedLoss", "MinCutLoss", "OrthoLoss"]
    with open(filename_0, "w", newline='') as f0:
        f0_csv = csv.writer(f0)
        f0_csv.writerow(headers_0)

    previous_loss = float("inf")  #Initialization.
    for epoch in range(1, Num_Epoch+1):  #Specify the number of epoch in each independent run.
        # 【修改点2】接收三个返回值
        train_loss, mc, o = train(epoch)
        # 每100个epoch打印一次，观察数值
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}, Total: {train_loss:.4f}, MC: {mc:.4f}, Ortho: {o:.4f}")

        #print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
        # 【修改点3】写入 CSV 时也要把 mc 和 o 写进去，不然和 Header 对不上
        with open(filename_0, "a", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow([epoch, train_loss, mc, o])
        
        if train_loss == 0 and train_loss == previous_loss:  #If two consecutive losses are both zeros, the learning gets stuck.
            break  #stop the training.
        else:
            previous_loss = train_loss

    print(f"Final train loss is {train_loss:.4f}")
    # 逻辑修改：不再强制因为 -0.6 而重试，改为只要不发散（比如 Loss > 10）就保存
    # 如果 Loss 异常大，可能是梯度爆炸，才需要重试
    if train_loss > 5.0:   #This is an empirical cutoff of the final loss to avoid underfitting.
        print("Loss too high (Diverged), retrying this run...")
        shutil.rmtree(RunFolderName)  #Remove the specific folder and all files inside it for re-creating the Run folder.
        continue  #restart this run.

    model.eval() # 切换到评估模式
    #Extract the soft TCN assignment matrix using the trained model.
    for EachData in all_sample_loader:
        EachData = EachData.to(device)
        #传入稀疏参数
        with torch.no_grad(): # 推理不需要梯度
            TestModelResult = model(EachData.x, EachData.edge_index, EachData.batch)

        ClusterAssignMatrix1 = TestModelResult[3]
        ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)  #Checked, consistent with the built-in function "dense_mincut_pool".
        ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().numpy() # 转回CPU
        filename1 = RunFolderName + "/TCN_AssignMatrix1.csv"
        np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

        ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
        ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().numpy()
        filename2 = RunFolderName + "/TCN_AdjMatrix1.csv"
        np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

        # NodeMask = EachData.mask
        # NodeMask = np.array(NodeMask)
        num_nodes = EachData.x.size(0)
        NodeMask = np.ones((num_nodes,), dtype=int)
        filename3 = RunFolderName + "/NodeMask.csv"
        np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')  #save as integers.

    run_number = run_number + 1

print("All runs finished!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


