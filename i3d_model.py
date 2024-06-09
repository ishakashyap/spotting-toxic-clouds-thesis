import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception3DModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception3DModule, self).__init__()
        assert len(out_channels) == 6
        
        self.branch1 = nn.Conv3d(in_channels, out_channels[0], kernel_size=1)
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[1], kernel_size=1),
            nn.Conv3d(out_channels[1], out_channels[2], kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[3], kernel_size=1),
            nn.Conv3d(out_channels[3], out_channels[4], kernel_size=3, padding=1)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels[5], kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class Inception3D(nn.Module):
    def __init__(self, num_classes=2):
        super(Inception3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv3d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = Inception3DModule(192, [64, 96, 128, 16, 32, 32])
        self.inception3b = Inception3DModule(256, [128, 128, 192, 32, 96, 64])
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception3DModule(480, [192, 96, 208, 16, 48, 64])
        # self.inception4b = Inception3DModule(512, [160, 112, 224, 24, 64, 64])
        # self.inception4c = Inception3DModule(512, [128, 128, 256, 24, 64, 64])
        # self.inception4d = Inception3DModule(512, [112, 144, 288, 32, 64, 64])
        # self.inception4e = Inception3DModule(528, [256, 160, 320, 32, 128, 128])
        # self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        
        # self.inception5a = Inception3DModule(832, [256, 160, 320, 32, 128, 128])
        # self.inception5b = Inception3DModule(832, [384, 192, 384, 48, 128, 128])
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        # x = self.inception4b(x)
        # x = self.inception4c(x)
        # x = self.inception4d(x)
        # x = self.inception4e(x)
        # x = self.maxpool4(x)
        
        # x = self.inception5a(x)
        # x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x