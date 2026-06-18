import torch
import torch.nn as nn
from .custom_cpp_ops import FastSwish

class RotationSolverNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Сверточная нейросеть (CNN) для обработки картинок (капчи) 32x32
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            FastSwish(), # Используем нашу C++ функцию активации!
            nn.MaxPool2d(2), # Картинка уменьшается до 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            FastSwish(),
            nn.MaxPool2d(2), # Картинка уменьшается до 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            FastSwish(),
            nn.MaxPool2d(2), # Картинка уменьшается до 4x4
        )
        
        # Регрессор: угадывает угол наклона животного
        self.regressor = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Выдает два числа: Синус и Косинус угла наклона
            nn.Linear(256, 2) 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.regressor(x)
        # Нормализуем, чтобы получить правильные координаты по кругу (вектор)
        return torch.nn.functional.normalize(out, p=2, dim=1)
