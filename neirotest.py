import llvmlite.ir as ir
import llvmlite.binding as llvm
import numpy as np
import array
import platform
import time
from numba import njit, prange, config
from ctypes import CFUNCTYPE, c_double, POINTER, c_int32

# 1. УНИВЕРСАЛЬНАЯ НАСТРОЙКА LLVM
try:
    llvm.initialize()
except RuntimeError:
    pass 

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvm.initialize_all_targets()
llvm.initialize_all_asmprinters()

def create_universal_engine(module):
    target_triple = llvm.get_process_triple() 
    target = llvm.Target.from_triple(target_triple)
    target_machine = target.create_target_machine(opt=3, features='', cpu='')
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine

def compile_ultra_fast_code():
    module = ir.Module(name="global_ai_core")
    vec_ty = ir.VectorType(ir.DoubleType(), 8)
    func_ty = ir.FunctionType(ir.DoubleType(), [ir.PointerType(ir.DoubleType()), ir.IntType(32)])
    func = ir.Function(module, func_ty, name="fast_solve")
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    ptr, length = func.args
    v_ptr = builder.bitcast(ptr, ir.PointerType(vec_ty))
    vector_val = builder.load(v_ptr)
    res_vec = builder.fmul(vector_val, ir.Constant(vec_ty, [1.0000001]*8))
    res = builder.extract_element(res_vec, ir.Constant(ir.IntType(32), 0))
    builder.ret(res)
    mod = llvm.parse_assembly(str(module))
    engine = create_universal_engine(mod)
    engine.add_module(mod)
    engine.finalize_object()
    addr = engine.get_function_address("fast_solve")
    return CFUNCTYPE(c_double, POINTER(c_double), c_int32)(addr)

_fast_func = compile_ultra_fast_code()

# 2. NUMBA: МНОГОПОТОЧНЫЕ ДВИЖКИ
config.THREADING_LAYER = 'omp' 

@njit(parallel=True, fastmath=True)
def activation_only(data_view):
    n = len(data_view)
    res = 0.0
    for i in prange(n):
        val = data_view[i]
        res += (np.exp(val) / (1.0 + np.exp(val)))
    return res

@njit(parallel=True, fastmath=True)
def dot_product_only(vec_a, vec_b):
    n = len(vec_a)
    res = 0.0
    for i in prange(n):
        res += vec_a[i] * vec_b[i]
    return res

# ТОТ САМЫЙ ОБЪЕДИНЕННЫЙ НЕЙРО-СЛОЙ (Fusion)
@njit(parallel=True, fastmath=True)
def full_neuro_layer(weights, inputs):
    n = len(weights)
    res = 0.0
    for i in prange(n):
        # Умножение весов на вход и тут же расчет активации
        z = weights[i] * inputs[i]
        res += (np.exp(z) / (1.0 + np.exp(z)))
    return res

# 3. ФИНАЛЬНЫЙ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ
def run_all_benchmarks():
    n = 100_000_000
    print(f"--- GLOBAL AI CORE: {platform.machine()} ---")
    
    # Данные для тестов
    weights = np.ones(n, dtype=np.float32)
    inputs = np.ones(n, dtype=np.float32)

    # Тест 1: Только Активация
    activation_only(weights[:1000]) # Прогрев
    start = time.time()
    activation_only(weights)
    print(f"[1/3] ТОЛЬКО АКТИВАЦИЯ: {time.time()-start:.6f} сек.")

    # Тест 2: Только Dot Product
    dot_product_only(weights[:1000], inputs[:1000]) # Прогрев
    start = time.time()
    dot_product_only(weights, inputs)
    print(f"[2/3] ТОЛЬКО DOT PRODUCT: {time.time()-start:.6f} сек.")

    # Тест 3: ПОЛНЫЙ НЕЙРО-СЛОЙ (Умножение + Активация)
    full_neuro_layer(weights[:1000], inputs[:1000]) # Прогрев
    start = time.time()
    full_neuro_layer(weights, inputs)
    dur = time.time() - start
    
    print(f"[3/3] ПОЛНЫЙ ЦИКЛ НЕЙРОНА: {dur:.6f} сек.")
    print(f"--- РЕЗУЛЬТАТ: {int(n/dur/1e6)} млн циклов в секунду ---")
    print("-" * 50)

if __name__ == "__main__":
    run_all_benchmarks()
