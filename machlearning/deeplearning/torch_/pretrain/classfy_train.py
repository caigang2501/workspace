import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

from torchvision.models import resnet50
import torch

import torch.nn as nn

num_classes = 3

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(CustomResNet50, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        label = int(os.path.basename(os.path.dirname(path)))
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

if __name__=='__main__':
    data_root = 'data/traindata/fire'

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    custom_dataset = CustomDataset(data_root, transform=transform)

    batch_size = 32
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    model = CustomResNet50(num_classes=num_classes)

    # Remove quantization
    # model = resnet50(pretrained=True)
    # model.load_state_dict(torch.load('path/to/your/unquantized_resnet50_weights.pth'))

    # Freeze some layers (optional)
    # for param in model.parameters():
    #     param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 1500
    min_loss = 0.4
    for epoch in range(num_epochs):
        for inputs, labels in custom_dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(dtype=torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        if loss.item()<min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), str(min_loss)+'torch_firedtc_resnet50.pth')







