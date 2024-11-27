import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_v2_s


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dec_hidden_size, backbone, transformation=None):

        super(EncoderCNN, self).__init__()

        self.embed_size = embed_size
        self.dec_hidden_size = dec_hidden_size
        self.backbone = backbone.lower()
        self.transformation = transformation.lower() if transformation else None

        #backbone model
        if self.backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048  #dimension of ResNet-50
            
            for param in resnet.parameters():
                param.requires_grad_(False)

            modules = list(resnet.children())[:-2]
            self.base_model = nn.Sequential(*modules)

        elif self.backbone == "efficientnet":
            efficientnet = efficientnet_v2_s(weights="IMAGENET1K_V1")
            self.feature_dim = 1280  #dimension of EfficientNet-V2-S

            for param in efficientnet.parameters():
                param.requires_grad_(False)

            self.base_model = nn.Sequential(*list(efficientnet.children())[:-2])

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        #Transformation layer
        if self.transformation == "conv2d":
            self.feature_transform = nn.Conv2d(self.feature_dim, dec_hidden_size, kernel_size=1)
        elif self.transformation is None:
            self.feature_transform = None
        else:
            raise ValueError(f"Unsupported transformation type: {self.transformation}")
        


    def forward(self, images):
        features = self.base_model(images)

        if self.feature_transform is not None:
            features = self.feature_transform(features)

        batch, feature_maps, size_1, size_2 = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, -1, feature_maps)

        return features
