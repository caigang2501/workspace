import os
from PIL import Image
import torch
import torch.nn as nn
from PIL import Image
from torchvision.io.image import read_image
from torchvision.models import resnet50
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights


class CustomDataset_pred(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        label = int(os.path.basename(os.path.dirname(path)))
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        resnet = resnet50(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Predictor:
    def __init__(self,model_path) -> None:
        self.transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = CustomResNet50(num_classes=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_bypath(self,data_root):
        custom_dataset = CustomDataset_pred(data_root, transform=self.transform)

        batch_size = 800
        custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

        for inputs, labels in custom_dataloader:
            with torch.no_grad():
                outputs = self.model(inputs)
            
            _, predicted = torch.max(outputs, 1)
        return _.numpy(),predicted.numpy()

    def predict(self,img_path):
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        img_transformed = img_transformed.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_transformed)

        predicted_class = torch.argmax(output, dim=1).item()

        # print("Predicted Class:", predicted_class)
        return predicted_class

def test():
    img = read_image(os.path.join(os.getcwd(),'data/traindata/imgs/bird1.jpg'))

    weights = ResNet50_QuantizedWeights.DEFAULT
    model = resnet50(weights=weights, quantize=True)
    model.eval()

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"categories: {category_name}: {100 * score}%")

if __name__=='__main__':
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/models/torch_firedtc_resnet50.pth')
    predictor = Predictor(model_path)

    # data_root = 'data/objdetect/train/fire'
    data_root = 'data/output_frames/frames/'
    result = predictor.predict_bypath(data_root)

    # bug exist
    # img_path = 'data/objdetect/train/1'
    # for file_name in os.listdir(img_path):
    #     file_path = os.path.join(img_path, file_name)
    #     print(predictor.predict(file_path))
    