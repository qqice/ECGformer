"""
模型导出脚本
将训练好的模型导出为不同格式，便于可视化和部署

支持的格式：
1. .keras - Keras 3.0+ 原生格式（推荐）
2. .h5 - HDF5格式（传统格式，兼容性好）
3. SavedModel - TensorFlow SavedModel格式（用于生产部署）
4. .onnx - ONNX格式（跨平台，便于分析运算需求）
"""

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

def export_model_formats(model_path, output_dir='./exported_models'):
    """
    将模型导出为多种格式

    参数:
        model_path: 训练好的模型路径
        output_dir: 导出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载模型: {model_path}")
    model = keras.models.load_model(model_path)

    # 1. 保存为 .keras 格式（推荐，Netron支持）
    keras_path = os.path.join(output_dir, 'ecgformer_model.keras')
    model.save(keras_path)
    print(f"✓ 已保存为 Keras格式: {keras_path}")

    # 2. 保存为 .h5 格式（HDF5格式，Netron支持）
    h5_path = os.path.join(output_dir, 'ecgformer_model.h5')
    model.save(h5_path, save_format='h5')
    print(f"✓ 已保存为 HDF5格式: {h5_path}")

    # # 3. 保存为 SavedModel 格式（TensorFlow原生格式）
    # savedmodel_path = os.path.join(output_dir, 'ecgformer_savedmodel')
    # model.save(savedmodel_path)
    # print(f"✓ 已保存为 SavedModel格式: {savedmodel_path}")

    # 4. 导出为ONNX格式
    try:
        import tf2onnx

        print("\n正在转换为ONNX格式...")
        onnx_path = os.path.join(output_dir, 'ecgformer_model.onnx')

        # 获取输入形状
        input_shape = model.input_shape
        print(f"模型输入形状: {input_shape}")

        # 方法1: 使用tf2onnx从SavedModel转换
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=13,  # ONNX opset版本
            output_path=onnx_path
        )

        print(f"✓ 已保存为 ONNX格式: {onnx_path}")

        # 显示ONNX模型信息
        print(f"\nONNX模型信息:")
        print(f"  - Opset版本: 13")
        print(f"  - 输入名称: {model.input.name}")
        print(f"  - 输入形状: {input_shape}")
        print(f"  - 输出名称: {model.output.name}")
        print(f"  - 输出形状: {model.output_shape}")

    except ImportError:
        print("\n⚠ 警告: 未安装tf2onnx库，无法导出ONNX格式")
        print("请运行: pip install tf2onnx")
        onnx_path = None
    except Exception as e:
        print(f"\n⚠ ONNX导出失败: {e}")
        onnx_path = None

    # 5. 显示模型信息
    print("\n" + "="*60)
    print("模型架构摘要:")
    print("="*60)
    model.summary()

    print("\n" + "="*60)
    print("导出完成！")
    print("="*60)
    print(f"你可以使用以下文件在Netron中可视化:")
    print(f"  - {keras_path}")
    print(f"  - {h5_path}")
    # print(f"  - {savedmodel_path}")
    if onnx_path:
        print(f"  - {onnx_path} (推荐用于运算分析)")

    print("\n使用ONNX的优势:")
    print("  ✓ 跨平台兼容性")
    print("  ✓ 详细的算子(operator)信息")
    print("  ✓ 便于分析计算复杂度(FLOPs)")
    print("  ✓ 可用于模型优化和部署")

    return model

if __name__ == "__main__":
    # 从checkpoint加载并导出模型
    checkpoint_path = './ckpts/best_model.keras'

    if os.path.exists(checkpoint_path):
        export_model_formats(checkpoint_path)
    else:
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        print("请先运行 train.py 训练模型")
