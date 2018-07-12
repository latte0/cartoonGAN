import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.AvgPool2d(3, 2, 1),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.AvgPool2d(3, 2, 1),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(2),
        )

    def forward(self, x):
        convs = self.convs(x)
        output = convs.view(convs.size(0), -1, 2)
        return output