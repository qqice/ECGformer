"""
ONNX模型分析脚本
分析模型的基础运算需求，包括FLOPs、参数量、算子类型等

需要安装的库:
pip install onnx onnxruntime
"""

import onnx
import numpy as np
from collections import Counter
import os

def analyze_onnx_model(onnx_path):
    """
    分析ONNX模型的详细信息

    参数:
        onnx_path: ONNX模型文件路径
    """
    if not os.path.exists(onnx_path):
        print(f"错误: 未找到ONNX模型文件 {onnx_path}")
        return

    print(f"正在加载ONNX模型: {onnx_path}")
    model = onnx.load(onnx_path)

    # 验证模型
    try:
        onnx.checker.check_model(model)
        print("✓ ONNX模型验证通过\n")
    except Exception as e:
        print(f"⚠ 模型验证警告: {e}\n")

    # 1. 基本信息
    print("="*70)
    print("【模型基本信息】")
    print("="*70)
    print(f"IR版本: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Graph名称: {model.graph.name}")
    print(f"Opset版本: {model.opset_import[0].version}")

    # 2. 输入输出信息
    print("\n" + "="*70)
    print("【输入输出信息】")
    print("="*70)

    print("\n输入:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        dtype = input_tensor.type.tensor_type.elem_type
        print(f"  - 名称: {input_tensor.name}")
        print(f"    形状: {shape}")
        print(f"    数据类型: {dtype}")

    print("\n输出:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        dtype = output_tensor.type.tensor_type.elem_type
        print(f"  - 名称: {output_tensor.name}")
        print(f"    形状: {shape}")
        print(f"    数据类型: {dtype}")

    # 3. 算子(Operator)分析
    print("\n" + "="*70)
    print("【算子(Operator)统计】")
    print("="*70)

    nodes = model.graph.node
    op_types = [node.op_type for node in nodes]
    op_counter = Counter(op_types)

    print(f"\n总节点数: {len(nodes)}")
    print(f"算子类型数: {len(op_counter)}")
    print("\n算子分布:")
    for op_type, count in sorted(op_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_type:20s}: {count:4d} 个")

    # 4. 参数量统计
    print("\n" + "="*70)
    print("【参数量统计】")
    print("="*70)

    total_params = 0
    initializers = model.graph.initializer

    print(f"\n可训练参数(Initializers)数量: {len(initializers)}")

    param_details = []
    for init in initializers:
        shape = list(init.dims)
        num_params = np.prod(shape) if shape else 1
        total_params += num_params
        param_details.append((init.name, shape, num_params))

    # 显示前10个最大的参数
    param_details.sort(key=lambda x: x[2], reverse=True)
    print("\n最大的10个参数张量:")
    for i, (name, shape, num) in enumerate(param_details[:10], 1):
        print(f"  {i:2d}. {name:40s} 形状:{str(shape):25s} 参数量:{num:>10,d}")

    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"参数大小(float32): {total_params*4/1024/1024:.2f} MB")

    # 5. 关键算子详细信息
    print("\n" + "="*70)
    print("【关键算子详细信息】")
    print("="*70)

    # 分析注意力机制相关的算子
    attention_ops = ['MatMul', 'Softmax', 'Add', 'Mul', 'Transpose']
    conv_ops = ['Conv', 'Conv1D', 'Conv2D']
    norm_ops = ['BatchNormalization', 'LayerNormalization', 'InstanceNormalization']

    print("\n注意力机制相关算子:")
    for op in attention_ops:
        if op in op_counter:
            print(f"  - {op}: {op_counter[op]} 个")

    print("\n卷积相关算子:")
    for op in conv_ops:
        if op in op_counter:
            print(f"  - {op}: {op_counter[op]} 个")

    print("\n归一化算子:")
    for op in norm_ops:
        if op in op_counter:
            print(f"  - {op}: {op_counter[op]} 个")

    # 6. 估算FLOPs（简化版）
    print("\n" + "="*70)
    print("【计算复杂度估算】")
    print("="*70)

    # 这是一个简化的FLOPs估算
    estimated_flops = 0

    # MatMul FLOPs估算 (2*M*N*K for M×K @ K×N)
    matmul_count = op_counter.get('MatMul', 0)
    if matmul_count > 0:
        print(f"\nMatMul操作数: {matmul_count}")
        print("  注: MatMul是Transformer模型的主要计算瓶颈")

    # Conv FLOPs
    conv_count = sum(op_counter.get(op, 0) for op in conv_ops)
    if conv_count > 0:
        print(f"\n卷积操作数: {conv_count}")

    print("\n注: 精确的FLOPs计算需要运行时的张量形状信息")
    print("建议使用专门的工具如 onnx-tool 或 thop 进行精确计算")

    # 7. 内存需求估算
    print("\n" + "="*70)
    print("【内存需求估算】")
    print("="*70)

    # 获取输入形状
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if None not in input_shape and -1 not in input_shape:
        input_size = np.prod(input_shape) * 4 / 1024  # float32, in KB
        print(f"\n单次推理输入大小: {input_size:.2f} KB")
        print(f"模型参数大小: {total_params*4/1024/1024:.2f} MB")
        print(f"预估总内存需求(推理): {total_params*4/1024/1024 + input_size/1024:.2f} MB")

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

def compare_with_standard_models():
    """显示与标准模型的对比"""
    print("\n" + "="*70)
    print("【与标准模型对比参考】")
    print("="*70)
    print("\n典型模型的参数量和FLOPs:")
    print("  ResNet-50:    ~25.6M参数,  ~4.1 GFLOPs")
    print("  VGG-16:       ~138M参数,   ~15.5 GFLOPs")
    print("  MobileNetV2:  ~3.5M参数,   ~0.3 GFLOPs")
    print("  BERT-Base:    ~110M参数,   ~22.5 GFLOPs")
    print("  ViT-Base:     ~86M参数,    ~17.6 GFLOPs")

if __name__ == "__main__":
    onnx_model_path = './exported_models/ecgformer_model.onnx'

    if os.path.exists(onnx_model_path):
        analyze_onnx_model(onnx_model_path)
        compare_with_standard_models()
    else:
        print(f"错误: 未找到ONNX模型文件")
        print(f"请先运行 export_model.py 导出模型")
        print("\n步骤:")
        print("  1. 确保已训练模型: python main_.py")
        print("  2. 导出模型: python export_model.py")
        print("  3. 分析模型: python analyze_onnx_model.py")

