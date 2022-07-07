import torch.nn as nn
import torch.nn.functional as F
import torch

# SpinalNet model architecture from https://github.com/dipuk0506/SpinalNet/blob/master/MNIST/SpinalNet_EMNIST_Digits.py
class SpinalNet(nn.Module):
    def __init__(self):
        super(SpinalNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 10) 
        self.fc1_1 = nn.Linear(160 + 10, 10) 
        self.fc1_2 = nn.Linear(160 + 10, 10) 
        self.fc1_3 = nn.Linear(160 + 10, 10) 
        self.fc1_4 = nn.Linear(160 + 10, 10) 
        self.fc1_5 = nn.Linear(160 + 10, 10) 
        self.fc2 = nn.Linear(10*6, 62) # changed from 10 to 62 because EMNISt has 62 classes
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x1 = x[:, 0:160]
        
        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,160:320], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:160], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,160:320], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:160], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,160:320], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

        x = self.fc2(x)
        y = F.log_softmax(x)
        return y
    