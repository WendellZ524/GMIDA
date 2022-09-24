import  torchvision.models.segmentation as models

class DeepLabV3():
    def __init__(self,backbone='resnet50', num_classes=19) -> None:
        self.latent_vector = None
    def __new__(self,backbone='resnet50', num_classes=19):
        return models.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

# if __name__ == "__main__":
#     import torch
#     x = torch.rand(1,3,128,256)
#     model = DeepLabV3()
#     y=model(x)['out']
#     print(y.size())

    
