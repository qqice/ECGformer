"""
验证C实现与TFLite模型的一致性

创建一个共享库接口，让Python可以调用C代码进行推理
"""

import numpy as np
import os
import sys
import ctypes
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
C_EXPORT_DIR = os.path.join(SCRIPT_DIR, 'c_export_modular')

# C代码的共享库版本
C_LIB_CODE = '''
// 添加到ecgformer_standalone.c来支持共享库接口

#ifdef BUILD_SHARED_LIB
// 导出函数
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void c_init(void) {
    init_tensors();
}

EXPORT int c_inference(const float* input, float* output) {
    return ecgformer_inference(input, output);
}

EXPORT void c_get_int8_output(int8_t* output) {
    ecgformer_get_int8_output(output);
}
#endif
'''


def compile_c_library():
    """编译C代码为共享库"""
    c_source = os.path.join(C_EXPORT_DIR, 'ecgformer_model.c')
    lib_path = os.path.join(C_EXPORT_DIR, 'libecgformer.so')
    
    if not os.path.exists(c_source):
        raise FileNotFoundError(f"C源文件不存在: {c_source}")
    
    # 添加共享库接口代码
    with open(c_source, 'r') as f:
        c_code = f.read()
    
    # 在main函数之前插入共享库接口
    if 'BUILD_SHARED_LIB' not in c_code:
        # 找到main函数位置
        main_pos = c_code.find('int main(int argc')
        if main_pos > 0:
            c_code = c_code[:main_pos] + C_LIB_CODE + '\n\n#ifndef BUILD_SHARED_LIB\n' + c_code[main_pos:]
            c_code += '\n#endif\n'
            
            with open(c_source, 'w') as f:
                f.write(c_code)
    
    # 编译为共享库
    cmd = [
        'gcc', '-O3', '-shared', '-fPIC',
        '-DBUILD_SHARED_LIB',
        '-o', lib_path,
        c_source,
        '-lm'
    ]
    
    print(f"编译命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"编译错误: {result.stderr}")
        raise RuntimeError("编译失败")
    
    print(f"共享库已编译: {lib_path}")
    return lib_path


class ECGformerC:
    """C实现的Python包装器"""
    
    def __init__(self):
        lib_path = os.path.join(C_EXPORT_DIR, 'libecgformer.so')
        
        if not os.path.exists(lib_path):
            print("共享库不存在，正在编译...")
            compile_c_library()
        
        self.lib = ctypes.CDLL(lib_path)
        
        # 设置函数签名
        self.lib.c_init.argtypes = []
        self.lib.c_init.restype = None
        
        self.lib.c_inference.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.c_inference.restype = ctypes.c_int
        
        self.lib.c_get_int8_output.argtypes = [ctypes.POINTER(ctypes.c_int8)]
        self.lib.c_get_int8_output.restype = None
        
        # 初始化
        self.lib.c_init()
    
    def inference(self, input_data: np.ndarray) -> tuple:
        """运行推理
        
        Args:
            input_data: 输入数据，形状 [187] 或 [1, 187, 1]
            
        Returns:
            (预测类别, 输出概率数组)
        """
        # 确保输入是正确的形状
        input_flat = input_data.flatten().astype(np.float32)
        if len(input_flat) != 187:
            raise ValueError(f"输入大小必须是187，得到 {len(input_flat)}")
        
        # 创建输出缓冲区
        output = np.zeros(5, dtype=np.float32)
        
        # 调用C函数
        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        pred = self.lib.c_inference(input_ptr, output_ptr)
        
        return pred, output
    
    def get_int8_output(self) -> np.ndarray:
        """获取INT8输出（用于验证）"""
        output = np.zeros(5, dtype=np.int8)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        self.lib.c_get_int8_output(output_ptr)
        return output


def verify_with_tflite():
    """验证C实现与TFLite的一致性"""
    import tensorflow as tf
    
    # 加载模型
    model_path = os.path.join(PROJECT_ROOT, 'exported_models', 'tflite', 
                              'ecgformer_custom_ln_int8.tflite')
    
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 获取输入量化参数
    input_scale = input_details[0]['quantization'][0]
    input_zp = input_details[0]['quantization'][1]
    
    # 初始化C模型
    c_model = ECGformerC()
    
    # 测试多个样本
    np.random.seed(42)
    n_tests = 100
    match_count = 0
    
    print(f"\n验证C实现与TFLite的一致性 ({n_tests}个测试样本)")
    print(f"输入量化参数: scale={input_scale}, zp={input_zp}")
    print("=" * 60)
    
    for i in range(n_tests):
        # 生成随机输入 (float)
        test_input_float = np.random.uniform(-0.5, 0.5, size=(1, 187, 1)).astype(np.float32)
        
        # 量化输入为INT8 (用于TFLite)
        test_input_int8 = np.round(test_input_float / input_scale + input_zp).astype(np.int8)
        
        # TFLite推理
        interpreter.set_tensor(input_details[0]['index'], test_input_int8)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]  # INT8
        
        # C推理 (C实现内部处理量化)
        pred, c_output_float = c_model.inference(test_input_float)
        c_output_int8 = c_model.get_int8_output()
        
        # 比较INT8输出
        matches = np.array_equal(tflite_output, c_output_int8)
        if matches:
            match_count += 1
        else:
            if i < 5:  # 只显示前5个不匹配的
                print(f"样本 {i}: 不匹配")
                print(f"  TFLite: {tflite_output}")
                print(f"  C:      {c_output_int8}")
                print(f"  差异:   {tflite_output - c_output_int8}")
    
    print(f"\n结果: {match_count}/{n_tests} 匹配 ({100*match_count/n_tests:.1f}%)")
    
    return match_count == n_tests


if __name__ == '__main__':
    # 先编译共享库
    try:
        compile_c_library()
    except Exception as e:
        print(f"编译失败: {e}")
        sys.exit(1)
    
    # 验证
    try:
        success = verify_with_tflite()
        if success:
            print("\n✓ C实现与TFLite完全匹配!")
        else:
            print("\n✗ C实现与TFLite存在差异")
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
