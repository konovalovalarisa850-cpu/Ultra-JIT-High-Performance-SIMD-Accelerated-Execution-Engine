import numpy as np
import time
import platform
from numba import njit, prange, config
import llvmlite.binding as llvm

# 1. СИСТЕМНАЯ ПОДГОТОВКА (Hardware Init)
config.THREADING_LAYER = 'omp'
try:
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
except:
    pass

# --- ГЛОБАЛЬНЫЙ НЕЙРО-ДВИЖОК (ПОЛНЫЙ ЦИКЛ) ---
class TurboFlashEngine:
    
    # АКТИВАЦИЯ + УМНОЖЕНИЕ (Forward Pass)
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def forward_layer(w, x, b):
        n = w.size
        out = np.empty(n, dtype=np.float32)
        for i in prange(n):
            # Loop Fusion: z = w*x + b -> sigmoid(z)
            z = w[i] * x[i] + b[i]
            out[i] = 1.0 / (1.0 + np.exp(-z))
        return out

    # ГРАДИЕНТНЫЙ СПУСК (Backward Pass / Training)
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def train_step(w, x, b, target, lr):
        n = w.size
        error_total = 0.0
        for i in prange(n):
            # Прямой ход
            z = w[i] * x[i] + b[i]
            pred = 1.0 / (1.0 + np.exp(-z))
            
            # Расчет ошибки (MSE)
            err = pred - target[i]
            error_total += err**2
            
            # ОБРАТНЫЙ ХОД (Backprop): вычисляем градиент сигмоиды
            # d_loss/d_w = err * sigmoid_der(z) * x
            gradient = err * (pred * (1.0 - pred))
            
            # ОБНОВЛЕНИЕ ВЕСОВ (Оптимизация на лету)
            w[i] -= lr * gradient * x[i]
            b[i] -= lr * gradient
            
        return error_total / n

# --- БИТВА С КОРПОРАЦИЯМИ (БЕНЧМАРК) ---
def run_global_test():
    size = 50_000_000 # 50 млн параметров
    print(f"--- TURBO FLASH ENGINE: {platform.processor()} ---")
    
    # Данные
    w = np.random.randn(size).astype(np.float32)
    x = np.random.randn(size).astype(np.float32)
    b = np.zeros(size, dtype=np.float32)
    target = np.random.rand(size).astype(np.float32)
    lr = 0.01

    engine = TurboFlashEngine()
    
    # 1. ТЕСТ СКОРОСТИ ОБУЧЕНИЯ (САМОЕ ВАЖНОЕ)
    engine.train_step(w[:100], x[:100], b[:100], target[:100], lr) # Прогрев
    
    print(f"[START] Запуск обучения на {size} параметрах...")
    t0 = time.perf_counter()
    
    loss = engine.train_step(w, x, b, target, lr)
    
    t1 = time.perf_counter()
    duration = t1 - t0
    
    print(f"--- РЕЗУЛЬТАТ ДЛЯ GOOGLE ---")
    print(f"Время одной эпохи: {duration:.4f} сек.")
    print(f"Скорость обучения: {int(size/duration/1e6)} МЛН ОПЕРАЦИЙ В СЕКУНДУ")
    print(f"Текущая ошибка (Loss): {loss:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    run_global_test()
