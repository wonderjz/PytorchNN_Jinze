# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SubNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),         # Adding batch normalization
            nn.LeakyReLU(),
#             nn.Dropout(0.3),  # Adding dropout
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),         # Adding batch normalization
#             nn.ReLU(),
#             nn.Dropout(0.5),            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),          # Adding batch normalization
            nn.LeakyReLU(),
#             nn.Dropout(0.3),
            nn.Linear(output_dim, 1)
        )
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)    
    def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        x = self.net(x)
        return x

class CombinedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,output_dimfinal, num_subnets):
        super(CombinedNetwork, self).__init__()
        self.num_subnets = num_subnets
        self.subnets = nn.ModuleList([
            SubNetwork(input_dim // num_subnets, hidden_dim, output_dim)
            for _ in range(num_subnets)
        ])
        self.compress_fc = nn.Linear(output_dim * num_subnets, output_dimfinal)
        self.final_fc = nn.Linear(output_dimfinal, output_dim)
        self.combine_fc = nn.Linear(num_subnets, output_dim)   
        # Loss function MSE
        self.criterion = nn.MSELoss(reduction='mean')
    def forward(self, x):
        # Calculate the remainder
        remainder = x.size(1) % self.num_subnets
        
        # If there is a remainder, pad the input tensor
        if remainder != 0:
            padding = torch.zeros(x.size(0), self.num_subnets - remainder).to(x.device)
            x = torch.cat([x, padding], dim=1)

        # Split the input into num_subnets parts
        x_split = torch.chunk(x, self.num_subnets, dim=1)
        
        # Pass each part through the corresponding sub-network
        outputs = [subnet(x_part) for subnet, x_part in zip(self.subnets, x_split)]
        
        # Concatenate the outputs of the sub-networks
        combined_output = torch.cat(outputs, dim=1)
        
        final_output = self.combine_fc(combined_output)
        return final_output.squeeze(1)
    
    
#         final_output = self.compress_fc(combined_output)
#         # Pass the concatenated output through the final fully connected layer
#         final_output = self.final_fc(final_output)
#         # Compute the mean of each row and reshape to batch_size by 1
#         final_output = final_output.mean(dim=1, keepdim=True) 
#         return final_output

    def cal_loss(self, pred, target, l2_lambda=1e-4):
        ''' Calculate loss with L2 regularization '''
        # Calculate the basic MSE loss
        mse_loss = self.criterion(pred, target)
        
#         # Calculate L2 regularization term (weight decay)
#         l2_reg = 0
#         for param in self.parameters():
#             l2_reg += torch.sum(param.pow(2))  # Sum of squared weights
#         mse_loss = mse_loss + l2_lambda * l2_reg
#         # Return the total loss: MSE loss + L2 regularization term
        return mse_loss
    
# # Example usage
# input_dim = 100  # Total number of input features
# hidden_dim = 256  # Hidden layer size in each sub-network
# output_dim = 1  # Output dimension
# num_subnets = 10  # Number of sub-networks

# # Create the combined network
# model = CombinedNetwork(input_dim, hidden_dim, output_dim, num_subnets)