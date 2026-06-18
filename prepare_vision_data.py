import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_data():
    print("Скачиваем датасет с животными и машинами (CIFAR-10)...")
    os.makedirs("dataset/vision_data", exist_ok=True)
    transform = transforms.ToTensor()
    
    # Автоматически качает 60,000 картинок
    trainset = torchvision.datasets.CIFAR10(root='./dataset/vision_data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./dataset/vision_data', train=False, download=True, transform=transform)
    
    print(f"Готово! Скачано {len(trainset) + len(testset)} картинок для капчи.")

if __name__ == "__main__":
    download_data()
