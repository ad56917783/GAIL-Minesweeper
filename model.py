import torch.nn as nn  
import torch
import torch.nn.functional as F  
import numpy as np



class PolicyNetwork(nn.Module):  
    def __init__(self):  
        super().__init__()  
        
        # 共享的特征提取器  
        self.feature_extractor = nn.Sequential(  
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 84x84 -> 42x42  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 42x42 -> 21x21  
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 21x21 -> 10x10  
            nn.Flatten()  
        )  
        
        # 计算卷积输出维度  
        self._compute_conv_output_dim()  
        
        # 位置预测头  
        self.position_head = nn.Sequential(  
            nn.Linear(self.conv_output_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, 2)  # 输出x,y坐标  
        )  
        
        # 动作类型预测头  
        self.action_head = nn.Sequential(  
            nn.Linear(self.conv_output_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, 4)  # 输出4种动作类型的概率  
        )  
    
    def _compute_conv_output_dim(self):  
        # 使用样例输入计算卷积层输出维度  
        x = torch.zeros(1, 1, 84, 84)  
        x = self.feature_extractor(x)  
        self.conv_output_dim = x.numel()  
    
    def forward(self, x):  
        # 确保输入维度正确  
        if len(x.shape) == 3:  
            x = x.unsqueeze(1)  # 添加通道维度  
        
        # 提取特征  
        features = self.feature_extractor(x)  
        
        # 预测位置和动作类型  
        positions = torch.sigmoid(self.position_head(features))  # 归一化到[0,1]  
        action_probs = F.softmax(self.action_head(features), dim=1)  # 动作概率  
        
        return positions, action_probs  

class Discriminator(nn.Module):  
    def __init__(self, input_dim=7062):  # 1*84*84 + 2 + 4 = 7062  
        super().__init__()  
        
        self.net = nn.Sequential(  
            nn.Linear(input_dim, input_dim*2),  
            nn.ReLU(),  
            nn.Linear(input_dim*2, 1),  
            nn.Sigmoid()
        )  
    
    def forward(self, x):  
        return self.net(x)  