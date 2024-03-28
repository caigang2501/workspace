import os

import torch
from PIL import Image
from torch import optim,nn
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import timm
from ellzaf_ml.models import GhostFaceNetsV2



# 模型配置
IMAGE_SIZE = 224
num_classes = 6
dic_names = {0:'NicoleKidman',1:'TangWei',2:'TaylorSwift',3:'WangZuXian',4:'ZhangGuoRong',5:'ZhangManYu'}

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        # 获取图像路径和标签
        path, _ = self.samples[index]
        # 解析类别标签
        label = int(os.path.basename(os.path.dirname(path)))
        img = self.loader(path)
        # 应用预处理
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def train():
    # 设置数据集路径
    data_root = 'machlearning/data/imgs/facedtc'

    # 定义数据预处理
    transform = T.Compose([
        T.Resize((224, 224)),  # 调整图像大小
        T.ToTensor(),           # 将图像转换为张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    custom_dataset = CustomDataset(data_root, transform=transform)

    batch_size = 32
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    # 创建自定义 ResNet50 模型
    model = GhostFaceNetsV2(image_size=IMAGE_SIZE, num_classes=num_classes, width=1, dropout=0.)
    
    # Remove quantization
    # model = resnet50(pretrained=True)
    # model.load_state_dict(torch.load('path/to/your/unquantized_resnet50_weights.pth'))

    # Freeze some layers (optional)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model on your custom dataset
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

    # Save the model
        if loss.item()<min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), str(min_loss)+'GhostFaceNet.pth')


def old_use_way():
    num_classes = 6

    img_folder = 'machlearning/data/facedtc'

    IMAGE_SIZE = 224
    # 统一的图像大小
    image_size = (224, 224)

    # 定义图像预处理操作
    preprocess = T.Compose([
        T.Resize(image_size),  # 调整图像大小为统一尺寸
        T.ToTensor(),  # 将图像转换为张量
    ])

    # 存储每张图片的张量
    img_tensors = []

    # 遍历图片文件夹
    for filename in os.listdir(img_folder):
        # 图片文件路径
        img_path = os.path.join(img_folder, filename)
        # 读取图片并进行预处理
        img = Image.open(img_path)
        img = img.convert('RGB')  # 将图片转换为 RGB 模式
        img_tensor = preprocess(img)
        # 将图片张量添加到列表中
        img_tensors.append(img_tensor)

    # 确保所有图片具有相同的形状
    min_height = min([img_tensor.shape[1] for img_tensor in img_tensors])
    min_width = min([img_tensor.shape[2] for img_tensor in img_tensors])

    # 裁剪或填充图片以使它们具有相同的大小
    preprocess_with_padding = T.Compose([
        T.CenterCrop((min_height, min_width)),  # 裁剪图片
        T.Resize(image_size),  # 调整图像大小为统一尺寸
    ])

    # 存储裁剪或填充后的图片张量
    img_tensors_final = []

    # 重新应用预处理操作
    for img_tensor in img_tensors:
        img_tensor_final = preprocess_with_padding(img_tensor)
        img_tensors_final.append(img_tensor_final)

    # 将图片张量堆叠成一个批次张量
    batch_img_tensor = torch.stack(img_tensors_final)
    print(batch_img_tensor.shape)

    # mobilenetv3 = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    # mobilenetv3.classifier = torch.nn.Linear(mobilenetv3.classifier.in_features, 2) #specify number of class here
    # model = LBPCNNFeatureFusion(backbone="mobilenetv3", adapt=True, backbone_model=mobilenetv3)
    model = GhostFaceNetsV2(image_size=IMAGE_SIZE, num_classes=num_classes, width=1, dropout=0.)

    preds = model(batch_img_tensor)
    print(preds)

if __name__=='__main__':
    train()