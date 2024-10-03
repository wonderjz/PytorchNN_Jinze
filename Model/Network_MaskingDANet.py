import torch
import torch.nn as nn
# Neural Network for tabular or say panel data
# simply replication and transformation from https://github.com/ChenJiahui777/DANet/blob/main/DANet.py
# Chen J, Liao K, Wan Y, et al. Danets: Deep abstract networks for tabular data classification and regression[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(4): 3930-3938.
class FixedMaskingModule(nn.Module):
    def __init__(self, num_features):
        super(FixedMaskingModule, self).__init__()
        self.num_features = num_features
        # 在初始化时创建一个固定的随机 mask
        self.mask = self.create_fixed_random_mask()

    def create_fixed_random_mask(self):
        num_masked = int(self.num_features * 0.1)  # 遮住十分之九的特征
        # 创建一个全为 False 的 mask
        mask = torch.zeros(self.num_features, dtype=torch.bool)
        # 随机选择要遮住的特征索引
        masked_indices = torch.randperm(self.num_features)[:num_masked]
        mask[masked_indices] = True  # 将选定的特征位置设置为 True
        return mask

    def forward(self, x):
        # 确保输入 x 不是 None
        if x is None:
            raise ValueError("Input tensor x is None.")
        

        # 确保 mask 的形状与输入 tensor 的最后一个维度匹配
        if x.size(1) != self.num_features:
            print(x.size(0))
            print(x.size(1))
            print(self.num_features)
            raise ValueError("Mask size does not match the number of features in the input tensor.")
        
        # 应用 mask 填充
        x = x.masked_fill(self.mask, 1e-6)
        return x

class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_out_dim):
        super(AbstractLayer, self).__init__()
        self.masker = FixedMaskingModule(num_features=base_input_dim)
        self.base_out_dim = base_out_dim
        # self.fc1 = nn.Sequential(ZScoreNormalize(), 
        self.fc1 = nn.Sequential(nn.Linear(base_input_dim, base_out_dim) )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc1(x)  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, input_dim, base_outdim, fix_input_dim):
        super(BasicBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim)
        self.conv2 = AbstractLayer(base_outdim, base_outdim)
        # self.downsample = nn.Sequential(
        #     #nn.Dropout(p=0.2),
        #     AbstractLayer(fix_input_dim, base_outdim)
        # )
        self.downsample = nn.Sequential(
            nn.Linear(fix_input_dim, base_outdim)
            # ,nn.ReLU()
            )
        self.modulerelu = nn.LeakyReLU()
        
    def forward(self, x, pre_out=None):
        if pre_out is None:
            pre_out = x
        # identity = x    
        identity = self.downsample(x)
        out = self.conv1(pre_out)
        #out = self.conv2(out)
        out = out + identity
        out1 = self.modulerelu(out)
        return out1

class DANet(nn.Module):
    def __init__(self, input_dim, num_classes, layer_num, base_outdim, fix_input_dim):
        super(DANet, self).__init__()
        params = {'base_outdim': base_outdim, 'fix_input_dim': fix_input_dim}
        self.init_layer = BasicBlock(fix_input_dim, **params)
        self.lay_num = layer_num
        self.layer = nn.ModuleList()
        for i in range(layer_num):
            self.layer.append(BasicBlock(base_outdim, **params))
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Sequential(nn.Linear(base_outdim, num_classes))
        # self.fc = nn.Sequential(nn.Linear(base_outdim, 8), nn.ReLU(), nn.Linear(8, num_classes))
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        out = self.init_layer(x)
        for i in range(len(self.layer)):
            out = self.layer[i](x, out)
        # out = self.drop(out)
        out = self.fc(out)
        return out

    def cal_loss(self, pred, target, lambda_reg=0.001, regularization_type='L1'):
        mse_loss = self.criterion(pred, target)
# Apply L1 regularization
        if regularization_type == None or 'L1':
            l1_norm = sum(p.abs().sum() for p in self.parameters())
#             l2_norm = sum(p.pow(2).sum() for p in self.parameters())
            mse_loss += lambda_reg * l1_norm
# Apply L2 regularization
        elif regularization_type == 'L2':
            l2_norm = sum(p.pow(2).sum() for p in self.parameters())
            mse_loss += lambda_reg * l2_norm
        regular_loss = mse_loss
        return regular_loss