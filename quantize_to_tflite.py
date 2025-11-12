"""
TFLite 模型量化脚本
支持两种量化方式：
1. 训练后量化 (Post-Training Quantization, PTQ)
   - Dynamic Range Quantization (动态范围量化)
   - Full Integer Quantization (全整数量化)
   - Float16 Quantization (Float16量化)
2. 量化感知训练 (Quantization-Aware Training, QAT)
   - 使用TensorFlow原生实现，无需额外依赖

使用方法:
    python quantize_to_tflite.py --method ptq --ptq_type dynamic
    python quantize_to_tflite.py --method ptq --ptq_type int8
    python quantize_to_tflite.py --method ptq --ptq_type float16
    python quantize_to_tflite.py --method qat --epochs 20
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 配置
MODEL_PATH = './ckpts/best_model.keras'
TRAIN_DATA_PATH = './dataset/mitbih_train.csv'
TEST_DATA_PATH = './dataset/mitbih_test.csv'
OUTPUT_DIR = './exported_models/tflite'


def readucr(filename):
    """读取并预处理数据"""
    data = pd.read_csv(filename, header=None)
    nRow, nCol = data.shape
    print(f'读取 {nRow} 行数据，{nCol} 列')
    y = data.iloc[:, -1].astype(int).to_numpy()
    x = data.iloc[:, :-1]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x)
    x = pd.DataFrame(standardized_data, columns=x.columns).to_numpy()
    return x, y.astype(int)


def load_data():
    """加载训练和测试数据"""
    print("\n" + "="*60)
    print("加载数据...")
    print("="*60)

    x_train, y_train = readucr(TRAIN_DATA_PATH)
    x_test, y_test = readucr(TEST_DATA_PATH)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    print(f"训练集形状: {x_train.shape}, 标签: {y_train.shape}")
    print(f"测试集形状: {x_test.shape}, 标签: {y_test.shape}")

    return x_train, y_train, x_test, y_test


def representative_dataset_gen(x_train, num_samples=100):
    """生成代表性数据集用于全整数量化"""
    def generator():
        # 随机选择样本
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        for i in indices:
            # 每次yield一个batch
            yield [x_train[i:i+1].astype(np.float32)]
    return generator


def post_training_quantization(model, x_train, ptq_type='dynamic'):
    """
    训练后量化 (PTQ)

    参数:
        model: 训练好的模型
        x_train: 训练数据（用于全整数量化的校准）
        ptq_type: 量化类型
            - 'dynamic': 动态范围量化（权重int8，激活float）
            - 'int8': 全整数量化（权重和激活都是int8）
            - 'float16': Float16量化
    """
    print("\n" + "="*60)
    print(f"执行训练后量化 (PTQ) - {ptq_type.upper()}")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 转换器基础设置
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if ptq_type == 'dynamic':
        # 动态范围量化：权重量化为int8，激活保持float32
        print("配置: 动态范围量化")
        print("  - 权重: int8")
        print("  - 激活: float32 (运行时动态量化)")
        print("  - 优点: 模型小，转换快，无需代表性数据")
        print("  - 缺点: 推理速度提升有限")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        output_file = 'ecgformer_ptq_dynamic.tflite'

    elif ptq_type == 'int8':
        # 全整数量化：权重和激活都量化为int8
        print("配置: 全整数量化")
        print("  - 权重: int8")
        print("  - 激活: int8")
        print("  - 优点: 模型最小，推理最快，支持整数算子")
        print("  - 缺点: 可能损失精度，需要代表性数据集")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(x_train, num_samples=100)

        # 确保输入输出都是int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        output_file = 'ecgformer_ptq_int8.tflite'

    elif ptq_type == 'float16':
        # Float16量化
        print("配置: Float16量化")
        print("  - 权重: float16")
        print("  - 激活: float16")
        print("  - 优点: 模型减半，精度损失小")
        print("  - 缺点: 不是所有硬件都支持加速")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        output_file = 'ecgformer_ptq_float16.tflite'

    else:
        raise ValueError(f"不支持的PTQ类型: {ptq_type}")

    # 执行转换
    print("\n开始转换模型...")
    tflite_model = converter.convert()

    # 保存模型
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # 显示模型大小
    model_size = len(tflite_model) / 1024
    print(f"\n✓ 量化完成！")
    print(f"  保存路径: {output_path}")
    print(f"  模型大小: {model_size:.2f} KB")

    return output_path


def quantization_aware_training(model_path, x_train, y_train, x_test, y_test,
                                epochs=10, batch_size=64):
    """
    量化感知训练 (QAT) - 简化版本
    使用较小学习率微调模型，然后应用PTQ

    注意：这是一个简化的QAT实现，不依赖tensorflow_model_optimization包。
    通过微调 + PTQ int8 的方式模拟QAT效果。

    参数:
        model_path: 预训练模型路径
        x_train, y_train: 训练数据
        x_test, y_test: 测试数据
        epochs: 微调轮数
        batch_size: 批次大小
    """
    print("\n" + "="*60)
    print("执行量化感知训练 (QAT) - 简化版本")
    print("="*60)
    print("说明:")
    print("  - 使用小学习率微调模型")
    print("  - 然后应用全整数PTQ量化")
    print("  - 不依赖tensorflow_model_optimization包")
    print(f"  - 微调轮数: {epochs}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载预训练模型
    print(f"\n加载预训练模型: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 评估原始模型
    print("\n评估原始模型性能:")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  原始模型 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # 克隆模型进行微调
    print("\n克隆模型进行微调...")

    # 使用非常小的学习率重新编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 微调训练
    print(f"\n开始微调训练 ({epochs} 轮)...")
    checkpoint_filepath = os.path.join(OUTPUT_DIR, 'qat_finetuned_model.keras')

    # 只使用部分数据进行快速微调
    train_size = min(len(x_train), 10000)  # 限制训练样本数量
    indices = np.random.choice(len(x_train), train_size, replace=False)
    x_train_subset = x_train[indices]
    y_train_subset = y_train[indices]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        x_train_subset,
        y_train_subset,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # 加载最佳模型
    finetuned_model = tf.keras.models.load_model(checkpoint_filepath)

    # 评估微调后模型
    print("\n评估微调后的模型:")
    ft_loss, ft_acc = finetuned_model.evaluate(x_test, y_test, verbose=0)
    print(f"  微调后模型 - Loss: {ft_loss:.4f}, Accuracy: {ft_acc:.4f}")

    # 应用PTQ量化（使用全整数量化）
    print("\n应用训练后量化...")
    converter = tf.lite.TFLiteConverter.from_keras_model(finetuned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(x_train, num_samples=100)

    # 使用全整数量化
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # 保存TFLite模型
    output_file = 'ecgformer_qat_int8.tflite'
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    model_size = len(tflite_model) / 1024
    print(f"\n✓ 简化QAT量化完成！")
    print(f"  保存路径: {output_path}")
    print(f"  模型大小: {model_size:.2f} KB")
    print(f"  原始精度: {acc:.4f}")
    print(f"  微调后精度: {ft_acc:.4f}")
    print(f"  精度变化: {(ft_acc - acc):.4f}")

    return output_path


def evaluate_tflite_model(tflite_path, x_test, y_test, num_samples=1000):
    """评估TFLite模型的性能"""
    print("\n" + "="*60)
    print(f"评估TFLite模型: {os.path.basename(tflite_path)}")
    print("="*60)

    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # 获取输入输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"输入详情: {input_details[0]}")
    print(f"输出详情: {output_details[0]}")

    # 检查是否需要量化输入
    input_scale = input_details[0].get('quantization_parameters', {}).get('scales')
    input_zero_point = input_details[0].get('quantization_parameters', {}).get('zero_points')
    output_scale = output_details[0].get('quantization_parameters', {}).get('scales')
    output_zero_point = output_details[0].get('quantization_parameters', {}).get('zero_points')

    is_quantized = input_details[0]['dtype'] == np.int8

    # 评估模型
    correct = 0
    total = min(num_samples, len(x_test))

    print(f"\n在 {total} 个样本上评估...")

    for i in range(total):
        # 准备输入
        test_input = x_test[i:i+1].astype(np.float32)

        # 如果是量化模型，需要量化输入
        if is_quantized and input_scale is not None:
            test_input = test_input / input_scale[0] + input_zero_point[0]
            test_input = test_input.astype(np.int8)

        # 设置输入
        interpreter.set_tensor(input_details[0]['index'], test_input)

        # 运行推理
        interpreter.invoke()

        # 获取输出
        output = interpreter.get_tensor(output_details[0]['index'])

        # 如果输出是量化的，需要反量化
        if is_quantized and output_scale is not None:
            output = (output.astype(np.float32) - output_zero_point[0]) * output_scale[0]

        # 预测类别
        pred = np.argmax(output[0])

        if pred == y_test[i]:
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{total} 样本...")

    accuracy = correct / total
    print(f"\n✓ TFLite模型精度: {accuracy:.4f} ({correct}/{total})")

    # 模型大小
    model_size = os.path.getsize(tflite_path) / 1024
    print(f"  模型大小: {model_size:.2f} KB")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description='ECGformer模型量化到TFLite')
    parser.add_argument('--method', type=str, choices=['ptq', 'qat'], required=True,
                        help='量化方法: ptq (训练后量化) 或 qat (量化感知训练)')
    parser.add_argument('--ptq_type', type=str, choices=['dynamic', 'int8', 'float16'],
                        default='dynamic',
                        help='PTQ类型: dynamic, int8, 或 float16')
    parser.add_argument('--epochs', type=int, default=10,
                        help='QAT微调轮数 (仅用于QAT)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (仅用于QAT)')
    parser.add_argument('--eval', action='store_true',
                        help='是否评估TFLite模型')
    parser.add_argument('--eval_samples', type=int, default=1000,
                        help='评估时使用的样本数')

    args = parser.parse_args()

    print("="*60)
    print("ECGformer TFLite 量化工具 (无需额外依赖)")
    print("="*60)

    # 加载数据
    x_train, y_train, x_test, y_test = load_data()

    # 加载原始模型
    print(f"\n加载原始模型: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 评估原始模型
    print("\n评估原始Keras模型:")
    original_loss, original_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Loss: {original_loss:.4f}, Accuracy: {original_acc:.4f}")

    # 获取模型大小（近似）
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=True) as tmp:
        model.save(tmp.name)
        original_size = os.path.getsize(tmp.name) / 1024
    print(f"  模型大小: {original_size:.2f} KB")

    # 执行量化
    if args.method == 'ptq':
        tflite_path = post_training_quantization(model, x_train, args.ptq_type)
    else:  # qat
        tflite_path = quantization_aware_training(
            MODEL_PATH, x_train, y_train, x_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size
        )

    # 评估TFLite模型
    if args.eval:
        tflite_acc = evaluate_tflite_model(tflite_path, x_test, y_test, args.eval_samples)

        print("\n" + "="*60)
        print("性能对比")
        print("="*60)
        print(f"原始模型精度: {original_acc:.4f}")
        print(f"TFLite模型精度: {tflite_acc:.4f}")
        print(f"精度变化: {(tflite_acc - original_acc):.4f}")

        tflite_size = os.path.getsize(tflite_path) / 1024
        print(f"\n原始模型大小: {original_size:.2f} KB")
        print(f"TFLite模型大小: {tflite_size:.2f} KB")
        print(f"压缩比: {(original_size / tflite_size):.2f}x")

    print("\n" + "="*60)
    print("量化完成！")
    print("="*60)
    print(f"\n输出文件: {tflite_path}")
    print("\n后续步骤:")
    print("  1. 使用以下命令评估模型性能:")
    print(f"     python quantize_to_tflite.py --method {args.method} {'--ptq_type ' + args.ptq_type if args.method == 'ptq' else ''} --eval")
    print("  2. 在移动设备或嵌入式设备上部署TFLite模型")
    print("  3. 使用TFLite推理引擎进行实时预测")


if __name__ == '__main__':
    main()
