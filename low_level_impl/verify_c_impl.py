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
import re
from typing import Dict, Any, Optional

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


def get_c_implementation_metrics(c_export_dir: str = None) -> Dict[str, Any]:
    """
    获取C实现的硬件相关指标
    
    解析生成的C头文件，提取以下信息：
    - 总参数量 (权重 + 偏置)
    - 参数存储大小 (字节)
    - 峰值激活内存
    - 内存槽配置
    - 模型结构信息
    
    Args:
        c_export_dir: C代码导出目录，默认为 c_export_modular
        
    Returns:
        包含各项指标的字典
    """
    if c_export_dir is None:
        c_export_dir = C_EXPORT_DIR
    
    metrics = {
        'weights': {'count': 0, 'size_bytes': 0, 'tensors': []},
        'biases': {'count': 0, 'size_bytes': 0, 'tensors': []},
        'memory': {},
        'model_config': {},
        'attention': {},
        'quantization': {}
    }
    
    # 1. 解析配置文件
    config_path = os.path.join(c_export_dir, 'ecgformer_config.h')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # 解析基本配置
        defines = {
            'INPUT_SIZE': r'#define\s+INPUT_SIZE\s+(\d+)',
            'OUTPUT_CLASSES': r'#define\s+OUTPUT_CLASSES\s+(\d+)',
            'NUM_TENSORS': r'#define\s+NUM_TENSORS\s+(\d+)',
            'ACTIVATION_POOL_SIZE': r'#define\s+ACTIVATION_POOL_SIZE\s+(\d+)',
            'NUM_MEMORY_SLOTS': r'#define\s+NUM_MEMORY_SLOTS\s+(\d+)',
            'MEMORY_LIMIT': r'#define\s+MEMORY_LIMIT\s+(\d+)',
            'NUM_HEADS': r'#define\s+NUM_HEADS\s+(\d+)',
            'SEQ_LEN': r'#define\s+SEQ_LEN\s+(\d+)',
            'HEAD_DIM': r'#define\s+HEAD_DIM\s+(\d+)',
            'ATTENTION_PER_HEAD': r'#define\s+ATTENTION_PER_HEAD\s+\(.*?\)\s*//\s*(\d+)',
        }
        
        for key, pattern in defines.items():
            match = re.search(pattern, config_content)
            if match:
                metrics['model_config'][key] = int(match.group(1))
        
        # 解析量化参数
        scale_match = re.search(r'#define\s+INPUT_SCALE\s+([\d.eE+-]+)f?', config_content)
        zp_match = re.search(r'#define\s+INPUT_ZERO_POINT\s+(-?\d+)', config_content)
        if scale_match:
            metrics['quantization']['input_scale'] = float(scale_match.group(1))
        if zp_match:
            metrics['quantization']['input_zero_point'] = int(zp_match.group(1))
        
        out_scale_match = re.search(r'#define\s+OUTPUT_SCALE\s+([\d.eE+-]+)f?', config_content)
        out_zp_match = re.search(r'#define\s+OUTPUT_ZERO_POINT\s+(-?\d+)', config_content)
        if out_scale_match:
            metrics['quantization']['output_scale'] = float(out_scale_match.group(1))
        if out_zp_match:
            metrics['quantization']['output_zero_point'] = int(out_zp_match.group(1))
        
        # 解析槽位大小
        slot_match = re.search(r'g_slot_sizes\[\d+\]\s*=\s*\{([^}]+)\}', config_content)
        if slot_match:
            slots = [int(x.strip()) for x in slot_match.group(1).split(',') if x.strip()]
            metrics['memory']['slot_sizes'] = slots
    
    # 2. 解析权重文件
    weights_path = os.path.join(c_export_dir, 'ecgformer_weights.h')
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            weights_content = f.read()
        
        # 匹配所有权重数组定义: static const int8_t weight_tXX[SIZE] = { ... };
        weight_pattern = r'//\s*张量\s*(\d+):\s*形状\s*\[([^\]]+)\]\s*\n\s*static\s+const\s+int8_t\s+weight_t\d+\[(\d+)\]'
        for match in re.finditer(weight_pattern, weights_content):
            tensor_id = int(match.group(1))
            shape_str = match.group(2)
            size = int(match.group(3))
            
            # 解析形状
            if shape_str.strip():
                shape = [int(x) for x in shape_str.split('x') if x.strip()]
            else:
                shape = [1]  # 标量
            
            metrics['weights']['tensors'].append({
                'id': tensor_id,
                'shape': shape,
                'size': size,
                'dtype': 'int8'
            })
            metrics['weights']['count'] += size
            metrics['weights']['size_bytes'] += size  # int8 = 1 byte
    
    # 3. 解析偏置文件
    biases_path = os.path.join(c_export_dir, 'ecgformer_biases.h')
    if os.path.exists(biases_path):
        with open(biases_path, 'r') as f:
            biases_content = f.read()
        
        # 匹配所有偏置数组定义: static const int32_t bias_tXX[SIZE] = { ... };
        bias_pattern = r'//\s*张量\s*(\d+):\s*形状\s*\[([^\]]*)\]\s*\n\s*static\s+const\s+int32_t\s+bias_t\d+\[(\d+)\]'
        for match in re.finditer(bias_pattern, biases_content):
            tensor_id = int(match.group(1))
            shape_str = match.group(2)
            size = int(match.group(3))
            
            # 解析形状
            if shape_str.strip():
                shape = [int(x) for x in shape_str.split('x') if x.strip()]
            else:
                shape = [1]  # 标量
            
            metrics['biases']['tensors'].append({
                'id': tensor_id,
                'shape': shape,
                'size': size,
                'dtype': 'int32'
            })
            metrics['biases']['count'] += size
            metrics['biases']['size_bytes'] += size * 4  # int32 = 4 bytes
    
    # 4. 计算汇总信息
    metrics['summary'] = {
        'total_parameters': metrics['weights']['count'] + metrics['biases']['count'],
        'weights_size_bytes': metrics['weights']['size_bytes'],
        'biases_size_bytes': metrics['biases']['size_bytes'],
        'total_params_size_bytes': metrics['weights']['size_bytes'] + metrics['biases']['size_bytes'],
        'activation_memory_bytes': metrics['model_config'].get('ACTIVATION_POOL_SIZE', 0),
        'num_weight_tensors': len(metrics['weights']['tensors']),
        'num_bias_tensors': len(metrics['biases']['tensors']),
    }
    
    # 计算总内存占用
    metrics['summary']['total_memory_bytes'] = (
        metrics['summary']['total_params_size_bytes'] + 
        metrics['summary']['activation_memory_bytes']
    )
    
    # 注意力模块信息
    if 'NUM_HEADS' in metrics['model_config']:
        metrics['attention'] = {
            'num_heads': metrics['model_config'].get('NUM_HEADS', 0),
            'seq_len': metrics['model_config'].get('SEQ_LEN', 0),
            'head_dim': metrics['model_config'].get('HEAD_DIM', 0),
            'per_head_attention_size': metrics['model_config'].get('SEQ_LEN', 0) ** 2,
            'full_attention_size': (metrics['model_config'].get('NUM_HEADS', 0) * 
                                    metrics['model_config'].get('SEQ_LEN', 0) ** 2),
        }
    
    return metrics


def print_c_implementation_metrics(c_export_dir: str = None, verbose: bool = True):
    """
    打印C实现的硬件相关指标
    
    Args:
        c_export_dir: C代码导出目录
        verbose: 是否显示详细信息
    """
    metrics = get_c_implementation_metrics(c_export_dir)
    
    print("\n" + "=" * 70)
    print("           ECGformer C实现 - 加速芯片相关指标")
    print("=" * 70)
    
    # 模型结构
    print("\n【模型结构】")
    print("-" * 50)
    cfg = metrics['model_config']
    print(f"  输入大小:        {cfg.get('INPUT_SIZE', 'N/A')} 个采样点")
    print(f"  输出类别:        {cfg.get('OUTPUT_CLASSES', 'N/A')} 类")
    print(f"  张量总数:        {cfg.get('NUM_TENSORS', 'N/A')} 个")
    
    # 参数统计
    print("\n【参数统计】")
    print("-" * 50)
    summary = metrics['summary']
    print(f"  权重张量数:      {summary['num_weight_tensors']} 个")
    print(f"  偏置张量数:      {summary['num_bias_tensors']} 个")
    print(f"  权重参数量:      {metrics['weights']['count']:,} 个 (INT8)")
    print(f"  偏置参数量:      {metrics['biases']['count']:,} 个 (INT32)")
    print(f"  总参数量:        {summary['total_parameters']:,} 个")
    
    # 存储需求
    print("\n【存储需求】")
    print("-" * 50)
    w_kb = summary['weights_size_bytes'] / 1024
    b_kb = summary['biases_size_bytes'] / 1024
    p_kb = summary['total_params_size_bytes'] / 1024
    print(f"  权重存储:        {summary['weights_size_bytes']:,} 字节 ({w_kb:.2f} KB)")
    print(f"  偏置存储:        {summary['biases_size_bytes']:,} 字节 ({b_kb:.2f} KB)")
    print(f"  参数总存储:      {summary['total_params_size_bytes']:,} 字节 ({p_kb:.2f} KB)")
    
    # 运行时内存
    print("\n【运行时内存 (激活值)】")
    print("-" * 50)
    act_bytes = summary['activation_memory_bytes']
    act_kb = act_bytes / 1024
    print(f"  峰值激活内存:    {act_bytes:,} 字节 ({act_kb:.2f} KB)")
    
    mem_cfg = metrics['memory']
    if 'slot_sizes' in mem_cfg:
        print(f"  内存槽数量:      {len(mem_cfg['slot_sizes'])} 个")
        print(f"  槽位大小分布:    {mem_cfg['slot_sizes']}")
        max_slot = max(mem_cfg['slot_sizes'])
        print(f"  最大槽位:        {max_slot:,} 字节 ({max_slot/1024:.2f} KB)")
    
    # 总内存
    print("\n【总内存需求】")
    print("-" * 50)
    total_bytes = summary['total_memory_bytes']
    total_kb = total_bytes / 1024
    print(f"  参数 + 激活:     {total_bytes:,} 字节 ({total_kb:.2f} KB)")
    
    limit = cfg.get('MEMORY_LIMIT', 262144)
    limit_kb = limit / 1024
    utilization = (act_bytes / limit) * 100 if limit > 0 else 0
    print(f"  内存限制:        {limit:,} 字节 ({limit_kb:.2f} KB)")
    print(f"  激活内存利用率:  {utilization:.1f}%")
    
    # 注意力模块
    if metrics['attention']:
        print("\n【注意力模块】")
        print("-" * 50)
        att = metrics['attention']
        print(f"  注意力头数:      {att['num_heads']}")
        print(f"  序列长度:        {att['seq_len']}")
        print(f"  每头维度:        {att['head_dim']}")
        print(f"  单头注意力大小:  {att['per_head_attention_size']:,} 字节")
        full_att_kb = att['full_attention_size'] / 1024
        print(f"  全注意力矩阵:    {att['full_attention_size']:,} 字节 ({full_att_kb:.2f} KB)")
        print(f"  → 优化策略: 逐Head计算，避免存储完整注意力矩阵")
    
    # 量化配置
    print("\n【量化配置】")
    print("-" * 50)
    quant = metrics['quantization']
    print(f"  输入 Scale:      {quant.get('input_scale', 'N/A')}")
    print(f"  输入 Zero Point: {quant.get('input_zero_point', 'N/A')}")
    print(f"  输出 Scale:      {quant.get('output_scale', 'N/A')}")
    print(f"  输出 Zero Point: {quant.get('output_zero_point', 'N/A')}")
    
    # 加速芯片关键指标汇总
    print("\n" + "=" * 70)
    print("           加速芯片部署关键指标汇总")
    print("=" * 70)
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Flash/ROM 需求 (参数存储):  {p_kb:>8.2f} KB            │")
    print(f"  │  SRAM 需求 (激活内存):       {act_kb:>8.2f} KB            │")
    print(f"  │  总内存需求:                 {total_kb:>8.2f} KB            │")
    print(f"  │  总参数量:                   {summary['total_parameters']:>8,} 个            │")
    print(f"  │  模型精度:                        INT8              │")
    print(f"  └─────────────────────────────────────────────────────┘")
    
    print()
    
    return metrics
    
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
    import argparse
    parser = argparse.ArgumentParser(description='验证C实现与TFLite模型的一致性')
    parser.add_argument('--metrics', '-m', action='store_true',
                        help='仅打印C实现的指标，不运行验证')
    parser.add_argument('--no-metrics', action='store_true',
                        help='不打印指标')
    args = parser.parse_args()
    
    if args.metrics:
        # 仅打印指标
        print_c_implementation_metrics()
    else:
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
        
        # 打印指标
        if not args.no_metrics:
            print_c_implementation_metrics()
