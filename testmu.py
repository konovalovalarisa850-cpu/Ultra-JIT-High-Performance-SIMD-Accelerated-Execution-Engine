import turbo_flash as tf
import numpy as np
import time

# Инициализируем данные для проверки (10 миллионов "нейронов")
size = 10_000_000
weights = np.random.rand(size).astype(np.float32)
inputs = np.random.rand(size).astype(np.float32)
bias = np.random.rand(size).astype(np.float32)

print("--- ЗАПУСК TURBO FLASH ENGINE ---")
start = time.time()

# ВЫЗОВ ТВОЕГО ДВИЖКА (используем метод accelerate_layer)
result = tf.engine.accelerate_layer(weights, inputs, bias)

end = time.time()
print(f"Время обработки 10 млн операций: {end - start:.6f} сек.")
print(f"Статус: УСПЕШНО (SIMD {tf.engine.vec_width} slots)")
