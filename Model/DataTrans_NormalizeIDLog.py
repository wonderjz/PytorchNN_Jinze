
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# Define a custom module that applies the log function
class LogModule(nn.Module):
    def __init__(self):
        super(LogModule, self).__init__()
    def forward(self, x):
        return torch.log(x + 1e-6)  # Add a small constant to avoid log(0)

class MinMaxNormalize(nn.Module):
    def __init__(self):
        super(MinMaxNormalize, self).__init__()
        self.register_buffer('min_val', torch.tensor(0.0))
        self.register_buffer('max_val', torch.tensor(1.0))

    def forward(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val + 1e-6)
    
class ZScoreNormalize(nn.Module):
    def __init__(self):
        super(ZScoreNormalize, self).__init__()
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-6)    

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self, x):
        return x
    #nn.Identity(x)





'''
class MYNeuralNet2(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, input_size4, input_size5, hidden_size, output_size):
        super(MYNeuralNet2, self).__init__()
        # module1 for nonlinear
        self.module1 = nn.Sequential(
            #MinMaxNormalize(),
            nn.Linear(input_size1, 64),
            #ZScoreNormalize(),
            #nn.softmax(),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            #LogModule(),
            nn.Linear(64, 64),
            #ZScoreNormalize(),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            #LogModule(),
            nn.Linear(64, 64)
        )
        # module2 for linear feature
        self.module2 = nn.Sequential(
            #Identity()
            nn.Linear(input_size2, 64),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(64, 64)
        )
        self.module3 = nn.Sequential(
            #Identity()
            nn.Linear(input_size3, 64),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(64, 64)
        )
        self.module4 = nn.Sequential(
            #Identity()
            nn.Linear(input_size4, 64),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(64, 64)
        )
        self.modulePCA = nn.Sequential(
            nn.Linear(5,output_size) # !!!!!
        )
        
        self.moduleID = nn.Identity()
        
        # fusion_module, the module 5
        self.module5 = nn.Sequential(
#             nn.Linear(5,5),
#             nn.LeakyReLU(),
            nn.Linear(64+64+64+64+93,1)
#             ,nn.LeakyReLU()
        )

        #self.fusion_layer = nn.Linear(92 + hidden_size1, output_size)
        # Loss function MSE
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
#         x_square2 = x ** 2
#         x_square3 = x ** 3
#         x_square4 = x ** 4
#         x1 = torch.cat((x[:],x_square2[:],x_square3[:],x_square3[:]), dim=1)
#         x2 = torch.cat((x[:],x_square2[:],x_square3[:],x_square3[:]), dim=1)
#         x1 = torch.cat((x[:],x_square2[:],x_square3[:]), dim=1)
#         x2 = torch.cat((x[:],x_square2[:]), dim=1)
        x1 = x[:]
        x2 = torch.cat((x[: ,40:44], x[: ,58:62], x[:, 76:80]), dim=1)
        x2 = x2[:]
        x3 = torch.cat((x[:, 44:52], x[: ,62:70], x[:, 80:88]), dim=1)
        x3 = x3[:]
        x4 = torch.cat((x[:, 52:57], x[: ,70:75], x[:, 88:93]), dim=1)
        x4 = x4[:]
        
#         xPCA = x[:]
#         # 将PyTorch张量转换为NumPy数组
#         data_numpy = xPCA.numpy()
#         # 初始化PCA对象，设置主成分数K
#         K = 5  # 例如，我们想要保留10个主成分
#         pca = PCA(n_components=K)
#         # 对数据进行PCA变换
#         # fit_transform方法会先拟合数据，然后将其转换为指定的主成分
#         transformed_data = pca.fit_transform(data_numpy)
#         # 将变换后的NumPy数组转换回PyTorch张量
#         xPCA = torch.tensor(transformed_data)


        out1 = self.module1(x1)
        out2 = self.module2(x2)
        out3 = self.module3(x3)
        out4 = self.module4(x4)
#         outPCA = self.modulePCA(xPCA)
#         identity_layer = nn.Identity()
#         out2 = identity_layer(x2)
        outID = self.moduleID(x[:])
        
        # 在特征维度上合并输出
        combined = torch.cat((out1,out2,out3,out4,outID), dim=1)
#         combined = torch.cat((out1,outPCA), dim=1)
        # 通过融合层得到最终输出
        final_output = self.module5(combined)
        #return self.net(x).squeeze(1)
        return final_output.squeeze(1)


    def cal_loss(self, pred, target,lambda_reg=0.1, regularization_type='L1'):
        # Calculate loss
        # You may try regularization here
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
        #return self.criterion(pred, target) 
        
# 假设表格数据有20列，前10列和后10列分别由不同的模块处理
input_size1 = 93 # !!!!!!!!!!!
input_size2 = 12
input_size3 = 24
input_size4 = 15
input_size5 = 93 # !!!!!

hidden_size = 1
output_size = 1   

'''