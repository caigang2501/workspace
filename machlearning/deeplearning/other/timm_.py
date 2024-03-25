import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

# 加载预训练的模型
# D:\anaconda3\Lib\site-packages\huggingface_hub\file_download.py
# C:\Users\EDY\.cache\huggingface\hub\models--timm--resnet50.a1_in1k
model = timm.create_model('resnet50', pretrained=True)

# 修改模型路径
# timm_checkpoint = 'hugface_models/pretrained_checkpoint.pth'
# model = timm.create_model('resnet50', pretrained=True, pretrained_model_checkpoint=timm_checkpoint)


# 读取并预处理图像
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open('machlearning/data/imgs/tangwei.jpg')
img = transform(img)
img = img.unsqueeze(0)  # 添加批次维度

# 设置模型为评估模式
model.eval()

# 前向传播
with torch.no_grad():
    output = model(img)

# 获取预测结果
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()

# 加载标签文件
with open('machlearning\data\imagenet-labels.txt') as f:
    labels = f.readlines()

# 打印预测结果
print(f"Predicted class: {labels[predicted_class]}")
print(f"Probability: {probabilities[predicted_class].item()}")
