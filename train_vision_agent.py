import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import random
from pathlib import Path
from tqdm import tqdm
from ai_brain.vision_net import RotationSolverNet

class RotationCaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    
    def __len__(self):
        return len(self.base)
        
    def __getitem__(self, idx):
        img, _ = self.base[idx] # Берем картинку (без оригинального класса животного)
        
        # Виртуально крутим картинку, имитируя hCaptcha/FunCaptcha
        angle = random.uniform(0, 360)
        rotated_img = transforms.functional.rotate(img, angle)
        
        # Нейросети проще предсказывать Синус и Косинус, чем просто угол в градусах
        angle_rad = math.radians(angle)
        label = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)
        
        return rotated_img, label

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучаем 'Убийцу Капчи' на {device}...")
    
    # 1. Загружаем и генерируем датасет (крученые картинки)
    base_train = torchvision.datasets.CIFAR10(root='./dataset/vision_data', train=True, download=False, transform=transforms.ToTensor())
    trainset = RotationCaptchaDataset(base_train)
    # Batch size 128 - картинки весят мало, поэтому батч огромный!
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    # 2. Создаем Мозг
    model = RotationSolverNet().to(device)
    criterion = nn.MSELoss() # Считаем ошибку (насколько градусов промахнулся ИИ)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    out_dir = Path("checkpoints/vision_agent")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Обучение
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Эпоха {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Эпоха {epoch+1} завершена. Ошибка: {running_loss/len(trainloader):.4f}")
        torch.save(model.state_dict(), out_dir / "vision_model.pt")
        print("Веса сохранены!")

if __name__ == "__main__":
    main()
