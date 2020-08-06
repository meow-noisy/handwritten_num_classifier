import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes, class_labels):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.class_labels = class_labels
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_probability(self, x):
        logit = self.forward(x)
        prob = self.softmax(logit)
        return prob

    def predict_labels(self, x):
        prob = self.predict_probability(x)
        values, indices = prob.max(1)
        
        confs = [v.data.item() for v in values]
        
        labels = [self.class_labels[idx.data.item()] for idx
                  in indices]
        
        
        return [labels, confs]


def load_alexnet(config):
    net = AlexNet(config['num_classes'],
                  config['class_labels'])
    
    return net

