import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.feature_dim = 2048  #ResNet-50 final feature dimension

        for param in resnet.parameters():
            param.requires_grad_(False)
        
        #Exclude the last two layers(Adaptive Average Pooling and Fully Connected)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        #Generates features with shape [batch_size, 2048, 7, 7], output of the last layer of resnet produces 7x7 feature maps
        #image_size / 32 = 224 / 32 = 7
        features = self.resnet(images)
        print("Encoder output shape:", features.shape)

        batch, feature_maps, size_1, size_2 = features.size()

        #[batch_size, 2048, 7, 7] -> [batch_size, 7, 7, 2048]
        features = features.permute(0, 2, 3, 1)

        #[batch_size, 7, 7, 2048] -> [batch_size, 49, 2048]
        features = features.view(batch, -1, feature_maps)
       
        return features 