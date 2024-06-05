import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils,datasets
# from keras import datasets
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # mnist = datasets.mnist.load_data()
    dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

    latent_dim = 100
    img_dim = 28 * 28

    generator = Generator(latent_dim, img_dim)
    discriminator = Discriminator(img_dim)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.view(imgs.size(0), -1)

            # 训练判别器
            optimizer_D.zero_grad()
            real_labels = torch.ones(imgs.size(0), 1)
            fake_labels = torch.zeros(imgs.size(0), 1)

            real_loss = criterion(discriminator(real_imgs), real_labels)
            z = torch.randn(imgs.size(0), latent_dim)
            fake_imgs = generator(z)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

        if epoch % 10 == 0:
            z = torch.randn(64, latent_dim)
            generated_imgs = generator(z).view(-1, 1, 28, 28)
            grid = utils.make_grid(generated_imgs, nrow=8, normalize=True)
            plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
            plt.show()

if __name__=='__main__':
    train()
