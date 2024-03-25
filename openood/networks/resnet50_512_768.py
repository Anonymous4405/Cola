from torchvision.models.resnet import Bottleneck, ResNet
import torch.nn as nn

class ResNet50_512(ResNet):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=1000):
        super(ResNet50_512, self).__init__(block=block,
                                       layers=layers,
                                       num_classes=num_classes)
        self.feature_size = 512
        self.fc1 = nn.Linear(2048, self.feature_size)
        self.fc = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        feature = self.fc1(feature)

        logits_cls = self.fc(feature)

        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc

    def get_feature(self, x):
        _, feature = self(x, return_feature=True)
        return feature

class ResNet50_768(ResNet):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=1000):
        super(ResNet50_768, self).__init__(block=block,
                                       layers=layers,
                                       num_classes=num_classes)
        self.feature_size = 768
        self.fc1 = nn.Linear(2048, self.feature_size)
        self.fc = nn.Linear(self.feature_size, num_classes)
        
    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        feature = self.fc1(feature)

        logits_cls = self.fc(feature)

        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc

    def get_feature(self, x):
        _, feature = self(x, return_feature=True)
        return feature

if __name__ == "__main__":
    import torch
    input = torch.rand(2, 3, 224, 224)
    model = ResNet50_512(num_classes=1000)
    output = model.get_feature(input)
    print(output.shape)