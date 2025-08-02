"""
FileName: 
Author: 
Version: 
Date: 2025/7/120:54
Description: 
"""
from torch.nn.utils.rnn import pad_sequence
import torch


def batch2pad(batch):
    '''
    The j-th element in batch vector is i if node j is in the i-th subgraph.
    The i-th row of pad matrix contains the nodes in the i-th subgraph.
    batch [0,1,0,0,1,1,2,2]->pad [[0,2,3],[1,4,5],[6,7,-1]]
    索引代表节点，值代表子图号，例如batch[0]=0,表示节点0在子图0中
    转换为pad，即pad=[0,2,3]，表示子图0中的节点有0,2,3号节点
    '''
    uni, inv = batch.unique(return_inverse=True)
    idx = torch.arange(inv.shape[0], device=batch.device)
    return pad_sequence([idx[batch == i] for i in uni[uni >= 0]],
                        batch_first=True,
                        padding_value=-1).to(torch.int64)


@torch.jit.script
def pad2batch(pad):
    '''
    pad [[0,2,3],[1,4,5],[6,7,-1]]->batch [0,1,0,0,1,1,2,2]
    这个方法就是batch2pad的逆操作，将pad转换为batch

    '''
    # 获取子图标签，例如pad.shape[0]= 3,得到[0,1,2]
    batch = torch.arange(pad.shape[0])
    # 变成列向量[[0], [1], [2]]
    batch = batch.reshape(-1, 1)
    # 广播，得到[[0,0,0],[1,1,1],[2,2,2]]
    batch = batch[:, torch.zeros(pad.shape[1], dtype=torch.int64)]
    # 变成一维的[0,0,0,1,1,1,2,2,2]
    batch = batch.to(pad.device).flatten()
    # 将输入的pad变成一维的[0,2,3,1,4,5,6,7,-1]
    pos = pad.flatten()
    # 标记出不为-1的元素，得到idx为
    idx = pos >= 0
    # 最终返回batch和pos分别是
    # batch [0,0,0,1,1,1,2,2]
    # pos [0,2,3,1,4,5,6,7]
    return batch[idx], pos[idx]


@torch.jit.script
def MaxZOZ(x, pos):
    '''
    produce max-zero-one label
    x is node feature
    pos is a pad matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph.
    -1 is padding value.
    产生max-zero-one标签，x是节点特征，pos是一个pad矩阵，其i-th行包含第i个子图中的节点。-1是填充值。

    将出现在pos矩阵中的节点对应标签设置为1，其余节点设置为0，得到z。即最大零一标签
    '''
    z = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
    pos = pos.flatten()
    # pos[pos >= 0] removes -1 from pos
    tpos = pos[pos >= 0].to(z.device)
    z[tpos] = 1
    return z
