import torch
import torchvision
import torchvision.transforms as transforms
from ai_brain.vision_net import RotationSolverNet
import math
from pathlib import Path
import random

def solve_captcha():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Загружаем обученную модель
    model = RotationSolverNet().to(device)
    ckpt = Path("checkpoints/vision_agent/vision_model.pt")
    if not ckpt.exists():
        print("Сначала обучите модель: python train_vision_agent.py")
        return
        
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    print("Модель-Решала загружена! Генерируем тестовую капчу...")
    
    # 2. Берем случайную картинку
    base_test = torchvision.datasets.CIFAR10(root='./dataset/vision_data', train=False, download=True, transform=transforms.ToTensor())
    img, _ = random.choice(base_test)
    
    # Крутим её на случайный градус (Имитация hCaptcha)
    secret_angle = random.uniform(0, 360)
    rotated_img = transforms.functional.rotate(img, secret_angle)
    
    print(f"\n[Сервер hCaptcha]: Картинка перевернута на {secret_angle:.1f} градусов. Поверните её ровно!")
    
    # 3. Отдаем перевернутую картинку нашей нейросети
    inputs = rotated_img.unsqueeze(0).to(device) # Добавляем batch-dimension
    
    with torch.no_grad():
        out = model(inputs)
        # Нейросеть выдает вектор [sin, cos]
        sin_pred, cos_pred = out[0].tolist()
        
        # Переводим обратно в градусы (Атангенс)
        predicted_angle_rad = math.atan2(sin_pred, cos_pred)
        predicted_angle = math.degrees(predicted_angle_rad) % 360
        
    print(f"[Наш ИИ 'Убийца капчи']: Я вижу картинку. Думаю, её крутили на {predicted_angle:.1f} градусов.")
    print(f"[Наш ИИ 'Убийца капчи']: Чтобы решить капчу, нужно повернуть ползунок на {-predicted_angle % 360:.1f} градусов.")
    
    error = min(abs(secret_angle - predicted_angle), 360 - abs(secret_angle - predicted_angle))
    print(f"-> Точность: ошибка всего {error:.1f} градусов!")
    
    if error < 15:
        print("\n✅ УСПЕХ! Капча пройдена (hCaptcha допускает ошибку до 20 градусов).")
    else:
        print("\n❌ ПРОВАЛ! Нужно больше эпох обучения.")

if __name__ == "__main__":
    solve_captcha()
