import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class Generator3DLUT(nn.Module):
    def __init__(self,dim=33,initialtype = 'zero'):
        super().__init__()
        if (initialtype == 'zero'):
            self.LUT = torch.zeros(3,dim,dim,dim,dtype = torch.float)
        else:
            self.LUT = torch.stack(torch.meshgrid(*[torch.arange(dim) for _ in range(3)]),dim=0).div(dim - 1).flip(0)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
    def forward(self,x):
        output = F.grid_sample(self.LUT.unsqueeze(0),x.unsqueeze(1),align_corners=True).squeeze(2)
        #print(output.shape)
        return output
        #return F.grid_sample(self.LUT.unsqueeze(0),x)
    
def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))
    return layers
class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.6),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)

class MergeWeight(nn.Module):
    def __ini__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 6, 5, stride=2, padding=1),
            nn.Conv2d(6,16,5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(16*5*5, 120),
            nn.InstanceNorm1d(120),
            nn.Linear(120,84),
            nn.Linear(84, 2),
            nn.InstanceNorm1d(2) 
        )
    
    def forward(self, img):
        return self.model(img)