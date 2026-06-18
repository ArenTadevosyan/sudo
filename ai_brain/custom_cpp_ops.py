import torch
from torch.utils.cpp_extension import load_inline

# Это чистый C++ код, который PyTorch скомпилирует в машинный код
cpp_source = """
#include <torch/extension.h>

// Прямой проход (Forward pass) нашей кастомной функции активации Swish
torch::Tensor fast_swish_forward(torch::Tensor x) {
    return x * torch::sigmoid(x);
}

// Обратный проход (Backward pass) для вычисления градиентов (производной)
torch::Tensor fast_swish_backward(torch::Tensor grad_output, torch::Tensor x) {
    auto sig = torch::sigmoid(x);
    return grad_output * (sig + x * sig * (1 - sig));
}
"""

print("Компилируем C++ код... Это займет пару секунд при первом запуске...")

# Компилируем C++ расширение прямо из Python (JIT компиляция)
custom_module = load_inline(
    name="custom_swish",
    cpp_sources=[cpp_source],
    functions=["fast_swish_forward", "fast_swish_backward"],
    verbose=False
)

# Оборачиваем C++ функции в слой PyTorch (Autograd Function)
class FastSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x) # Сохраняем X для вычисления градиента
        return custom_module.fast_swish_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Вызываем C++ функцию обратного распространения ошибки
        return custom_module.fast_swish_backward(grad_output, x)

# Создаем обычный слой нейросети (как nn.ReLU), но работающий на нашем C++ коде
class FastSwish(torch.nn.Module):
    def forward(self, x):
        return FastSwishFunction.apply(x)
