#导入依赖库
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pandas as pd
import math
import datetime
import os
import shutil
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset


# 超参数
InputFolderName = "./MIBI-TNBC_Input/"#输入数据的路径
KNN_K = 72#KNN的N


# 读取图像名称列表【这一步没理解，图像究竟是什么】
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
        Region_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["Image"],  # set our own names for the columns
    )


## 生成细胞空间图的拓扑结构
ThisStep_OutputFolderName = "./Step1_Output/"#输出文件夹路径
if os.path.exists(ThisStep_OutputFolderName):#如果文件夹已经存在了，那就删除旧的，创建新的文件
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)
#记录时间【注意，这里的图像指的是细胞地图，就是每个细胞的位置和细胞的类型】
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))#打印程序开始的时间
print("Constructing topology structures of KNN graphs...")#告诉庸俗正在做什么
for graph_index in range(0, len(region_name_list)):#遍历每个图像

    print(f"This is image-{graph_index}")
    # 获取当前要处理的图片名字
    region_name = region_name_list.Image[graph_index]
    # 读取该图片的“坐标文件”
    GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
    x_y_coordinates = np.loadtxt(GraphCoord_filename, dtype='float', delimiter="\t")

    # 计算邻居
    K = KNN_K
    # 找到距离最近72个邻居，只记录连没连接（0或者1），自己不能是自己的邻居
    KNNgraph_sparse = kneighbors_graph(x_y_coordinates, K, mode='connectivity', include_self=False, n_jobs=-1)  
    KNNgraph_AdjMat = KNNgraph_sparse.toarray()
    # 变成对称图，这样一想加（矩阵+矩阵的转置）
    KNNgraph_AdjMat_fix = KNNgraph_AdjMat + KNNgraph_AdjMat.T  #2min and cost one hundred memory.
    # 提取并保留
    KNNgraph_EdgeIndex = np.argwhere(KNNgraph_AdjMat_fix > 0)  #1min
    filename0 = ThisStep_OutputFolderName + region_name + "_EdgeIndex.txt"
    #保存成txt文件
    np.savetxt(filename0, KNNgraph_EdgeIndex, delimiter='\t', fmt='%i')  #save as integers. Checked the bidirectional edges.
    
print("All topology structures have been generated!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


## 制作细胞身份证
print("Generating node attribute matrices of KNN graphs...")
cell_type_vec = []#保存细胞类型
num_nodes = []#保存每个图多少个细胞
for graph_index in range(0, len(region_name_list)):

    # 读取文件名
    region_name = region_name_list.Image[graph_index]
    # 读取细胞类型
    CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
    cell_type_label = pd.read_csv(
        CellType_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["cell_type"],  # set our own names for the columns
    )
    # 把每一张图图里面所有的细胞类型添加到大列表里面
    cell_type_vec.extend(cell_type_label["cell_type"].values.tolist())
    num_nodes.append(len(cell_type_label))#记一下有多少个细胞

#去重，记录细胞类型
cell_type_vec_uniq = list(set(cell_type_vec))  # generate a vector of unique cell types and store it to .txt for final illustration.
#保存这个列表
CellTypeVec_filename = ThisStep_OutputFolderName + "UniqueCellTypeList.txt"
with open(CellTypeVec_filename, 'w') as fp:
    for item in cell_type_vec_uniq:
        # write each item on a new line
        fp.write("%s\n" % item)

# 找出最大的图有多大（现存爆炸的原因，让所有的图一样大）
max_nodes = math.ceil(max(num_nodes))  # generate the max number of cells and store this value to .txt for the next step.
MaxNumNodes_filename = ThisStep_OutputFolderName + "MaxNumNodes.txt"
with open(MaxNumNodes_filename, 'w') as fp1:
    fp1.write("%i\n" % max_nodes)

# 生成数字矩阵，生成one-hot标记
for graph_index in range(0, len(region_name_list)):

    print(f"This is image-{graph_index}")
    region_name = region_name_list.Image[graph_index]
    # import cell type label.
    CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
    cell_type_label = pd.read_csv(
        CellType_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["cell_type"],  # set our own names for the columns
    )

    # 初始化一个全是0的矩阵：行数=细胞数，列数=细胞种类数
    node_attr_matrix = np.zeros((len(cell_type_label), len(cell_type_vec_uniq)))
    for cell_ind in range(0, len(cell_type_label)):
        # 如果第3个细胞是“类型B”，就在第3行、“类型B”对应的那一列填 1
        type_index = cell_type_vec_uniq.index(cell_type_label["cell_type"][cell_ind])
        node_attr_matrix[cell_ind, type_index] = 1  # make the one-hot vector for each cell.

    #保存矩阵
    filename1 = ThisStep_OutputFolderName + region_name + "_NodeAttr.txt"
    np.savetxt(filename1, node_attr_matrix, delimiter='\t', fmt='%i')  #save as integers.

print("All node attribute matrices have been generated!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


## 到爆成深度学习可以用的数据包
print("Start graph data structure transformation...")
# Construct ordinary Python list to hold all input graphs.
data_list = []
for i in range(0, len(region_name_list)):
    region_name = region_name_list.Image[i]

    # 1读回“连线表” (Edge Index)
    EdgeIndex_filename = ThisStep_OutputFolderName + region_name + "_EdgeIndex.txt"
    edge_ndarray = np.loadtxt(EdgeIndex_filename, dtype = 'int64', delimiter = "\t")
    # 转换成 PyTorch 的 Tensor 格式
    edge_index = torch.from_numpy(edge_ndarray)
    #print(edge_index.type()) #should be torch.LongTensor due to its dtype=torch.int64

    # 2读回“身份证” (Node Attribute)
    NodeAttr_filename = ThisStep_OutputFolderName + region_name + "_NodeAttr.txt"
    x_ndarray = np.loadtxt(NodeAttr_filename, dtype='float32', delimiter="\t")  #should be float32 not float or float64.
    x = torch.from_numpy(x_ndarray)
    #print(x.type()) #should be torch.FloatTensor not torch.DoubleTensor.
    
    # 3读取这张图的标签 (是病图还是健康图？)
    GraphLabel_filename = InputFolderName + region_name + "_GraphLabel.txt"
    graph_label = np.loadtxt(GraphLabel_filename, dtype = 'int64', delimiter="\t")  #change to int64 from int due to expected torch.LongTensor.
    y = torch.from_numpy(graph_label)
    #print(y.type()) #should be torch.LongTensor due to its dtype=torch.int64

    # 4打包成一个 Data 对象
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    data_list.append(data)

# Define "SpatialOmicsImageDataset" class based on ordinary Python list.
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

# Create an object of this "SpatialOmicsImageDataset" class.
#下面是你要修改的代码
#dataset = SpatialOmicsImageDataset(ThisStep_OutputFolderName, transform=T.ToDense(max_nodes))
dataset = SpatialOmicsImageDataset(ThisStep_OutputFolderName)
print("Step1 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


