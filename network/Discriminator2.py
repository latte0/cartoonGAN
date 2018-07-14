import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()


        '''
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
        '''

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 32, 3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2 ,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3),
            InstanceNormalization(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, 2 ,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3),
            InstanceNormalization(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3),
            InstanceNormalization(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 3),


        )

    def forward(self, x):
        convs = self.convs(x)
        #output = convs.view(convs.size(0), -1, 2)
        return convs


class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def __call__(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out



