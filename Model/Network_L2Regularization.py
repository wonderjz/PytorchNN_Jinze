import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        
        # Try to modify this DNN to achieve better performance
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),         # Adding batch normalization
            nn.ReLU(),
#             nn.Dropout(0.3),  # Adding dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),          # Adding batch normalization
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        

        # Loss function MSE
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target, l2_lambda=1e-5):
        ''' Calculate loss with L2 regularization '''
        # Calculate the basic MSE loss
        mse_loss = self.criterion(pred, target)
        
        # Calculate L2 regularization term (weight decay)
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param.pow(2))  # Sum of squared weights
        mse_loss = mse_loss + l2_lambda * l2_reg
        # Return the total loss: MSE loss + L2 regularization term
        return mse_loss
