import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.amp import GradScaler, autocast  # Updated import
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import spectral_norm

# 优化 GPU 性能
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# 自定义数据集
class CatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
        if len(self.images) == 0:
            raise ValueError(f"没有在 {data_dir} 中找到图片。请检查路径或文件扩展名。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"加载图片 {img_path} 出错：{e}")
        if self.transform:
            image = self.transform(image)
        return image

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 512 * 4 * 4),
            nn.ReLU(True),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 判别器（移除 Sigmoid，因为 BCEWithLogitsLoss 包含 Sigmoid）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1)  # 移除 Sigmoid
        )

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_gan(dataloader, epochs, device):
    scaler = GradScaler('cuda')  # Updated to torch.amp.GradScaler
    fixed_noise = torch.randn(16, 100, device=device)
    for epoch in range(epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device, memory_format=torch.channels_last)
            batch_size = real_images.size(0)

            # 添加噪声到真实和生成图片
            noise_real = torch.randn_like(real_images) * 0.05
            noise_fake = torch.randn(batch_size, 3, 64, 64, device=device) * 0.05

            # 训练判别器
            discriminator.zero_grad()
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # 标签平滑
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # 标签平滑
            with autocast('cuda'):  # Updated to torch.amp.autocast
                real_output = discriminator(real_images + noise_real)
                d_loss_real = criterion(real_output, real_labels)
                noise = torch.randn(batch_size, 100, device=device)
                fake_images = generator(noise)
                fake_output = discriminator((fake_images.detach() + noise_fake))
                d_loss_fake = criterion(fake_output, fake_labels)
                d_loss = d_loss_real + d_loss_fake
            scaler.scale(d_loss).backward()
            scaler.step(discriminator_optimizer)
            scaler.update()

            # 训练生成器
            generator.zero_grad()
            with autocast('cuda'):  # Updated to torch.amp.autocast
                fake_output = discriminator(fake_images)
                g_loss = criterion(fake_output, real_labels)
            scaler.scale(g_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            # 更新学习率
            scheduler_g.step()
            scheduler_d.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

        # 每10个epoch保存生成图片
        if (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_noise).cpu()
                fake_images = (fake_images + 1) / 2  # 归一化到 [0, 1]
                vutils.save_image(fake_images, f"image_at_epoch_{epoch+1:04d}.png", nrow=4, normalize=True)
            generator.train()

if __name__ == '__main__':
    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

    # 数据加载
    data_dir = r'./cats/Data'
    batch_size = 64  # 增加批量大小（根据 GPU 内存调整）
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = CatDataset(data_dir, transform=transform)
        print(f"找到 {len(dataset)} 张图片")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    except ValueError as e:
        print(e)
        exit()

    # 初始化模型和优化器
    generator = Generator().to(device, memory_format=torch.channels_last)
    discriminator = Discriminator().to(device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss()  # Updated to BCEWithLogitsLoss
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 定义 epochs
    epochs = 100

    # 学习率调度器
    scheduler_g = CosineAnnealingLR(generator_optimizer, T_max=epochs)
    scheduler_d = CosineAnnealingLR(discriminator_optimizer, T_max=epochs)

    # 验证模型是否在 GPU 上
    print(f"生成器设备: {next(generator.parameters()).device}")
    print(f"判别器设备: {next(discriminator.parameters()).device}")

    # 运行训练
    train_gan(dataloader, epochs, device)

    # 保存最终生成图片
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1, 100, device=device)
        generated_image = generator(noise).cpu()
        generated_image = (generated_image + 1) / 2  # 归一化到 [0, 1]
        vutils.save_image(generated_image, "final_generated_image.png", normalize=True)
    generator.train()

    # 保存模型
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("模型和最终生成图片已保存")