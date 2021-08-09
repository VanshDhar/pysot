from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

class TEMPORAL(nn.Module):
    def __init__(self):
        super(TEMPORAL, self).__init__()

    def forward(self, temporal_images):
        raise NotImplementedError

class TemporalNet(TEMPORAL):
    def __init__(self):
        super(TemporalNet, self).__init__()
        # Input Shape = [1, 3, 255, 255, 50]
        # Output at Layer 2 of ResNet50 Block: [1, 1, 512]
        # Output at Layer 3 of ResNet50 Block: [1, 1, 1024]
        # Output at Layer 4 of ResNet50 Block: [1, 1, 2048]
        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 32, 3, 1, 1),
            nn.AvgPool3d([3,3,5], [1,1,5], 1), # Image shape = [1, 32, 255, 255, 10]
            nn.ReLU(),

            nn.Conv3d(32, 16, 3, 1, 1),
            nn.AvgPool3d([3,3,2], [1,1,2], 1), # Image shape = [1, 16, 255, 255, 6]
            nn.ReLU(),

            nn.Conv3d(16, 8, 3, 1, 1),
            nn.AvgPool3d([3, 3, 5], [1, 1, 5], 1), # Image Shape = [1, 8, 255, 255, 1]
            nn.ReLU(),

            nn.Conv3d(8, 1, 3, 1, 1), # Image Shape = [1, 1, 255, 255, 1]
            nn.ReLU(),

            nn.Conv3d(1, 64, 3, 2, 1),
            nn.ReLU(),

            nn.Conv3d(64, 128, 3, 2, 1),
            nn.ReLU(),

            nn.Conv3d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.Conv3d(256, 512, [3,3,1], 2, 0), # Image Shape = [1, 512, 32, 32, 1]
            nn.ReLU(),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(512, 1024, 3, 1, 1), # Image Shape = [1, 1024, 32, 32, 1]
            nn.ReLU(),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(1024, 2048, 3, 1, 1), # Image Shape = [1, 2048, 32, 32, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        out2 = self.Conv1(x)
        out3 = self.Conv2(out2)
        out4 = self.Conv3(out3)

        out = [out2, out3, out4]
        return out


# def TempNet():
#     model = TemporalNet()
#     return model



# if __name__ == '__main__':
#     net = TempNet()
#     # print(net)
#     net = net.cuda()
#     var = torch.FloatTensor(1, 3, 255, 255, 50).cuda()
#     # y = torch.FloatTensor(1, 512, 32, 32).cuda()
#     var = net(var)

#     for i in range(len(var)):
#         print(var[i].shape)

    # x = var[0]
    # x = torch.reshape(x, (1, 512, 32, 32))
    # x = x.add(y)
    # x = x + y
    # x = torch.cat((x, y), 1)
    # print(x.shape)
    # print(x)