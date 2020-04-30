import torch
import torch.nn as nn
import scipy.spatial

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.#特征维度
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))#生成10行2列的向量

    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)#x的size为(16,200)
        ###dismat为x和centers的欧氏距离。
        # .expand()返回tensor的一个新视图，单个维度扩大为更大的尺寸。.t()是转置
        #前面是把特征维度总和扩大为（batch_size,类别）
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t()) #dismat是[16,200]
        classes = torch.arange(self.num_classes).long()

        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#原先的labels是16，现在的是[16,200]
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #mask为每个标签的类别向量张量，即一行为一个[0,0,...,1,....]这样的类别向量。
        dist = distmat * mask.float()#保留下对应类的dismat中每行对应类别列，其他为0

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size #loss = dist中大于0的元素的和 / batch_size,想离中心越近越好

        return loss