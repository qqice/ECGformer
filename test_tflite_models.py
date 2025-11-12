"""
TFLite模型测试和对比脚本
在整个测试集上测试不同量化的TFLite模型，并与原始Keras模型对比
绘制性能对比图表

使用方法:
    python test_tflite_models.py
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 配置
KERAS_MODEL_PATH = './ckpts/best_model.keras'
TFLITE_DIR = './exported_models/tflite'
TEST_DATA_PATH = './dataset/mitbih_test.csv'
OUTPUT_DIR = './results'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def load_test_data():
    """加载测试数据"""
    print("\n" + "="*70)
    print("加载测试数据...")
    print("="*70)
    x_test, y_test = readucr(TEST_DATA_PATH)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    print(f"测试集形状: {x_test.shape}, 标签: {y_test.shape}")
    return x_test, y_test


def test_keras_model(model_path, x_test, y_test):
    """测试原始Keras模型"""
    print("\n" + "="*70)
    print("测试原始Keras模型")
    print("="*70)

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ 成功加载模型: {model_path}")

        # 测试推理时间
        start_time = time.time()
        predictions = model.predict(x_test, verbose=0)
        inference_time = time.time() - start_time

        y_pred = np.argmax(predictions, axis=1)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # 获取模型大小
        model_size = os.path.getsize(model_path) / 1024  # KB

        results = {
            'model_name': 'Keras (原始)',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'avg_inference_time': inference_time / len(x_test) * 1000,  # ms per sample
            'model_size_mb': model_size / 1024,  # Convert to MB
            'model_size_kb': model_size,
            'predictions': y_pred
        }

        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"总推理时间: {inference_time:.2f}秒")
        print(f"平均推理时间: {results['avg_inference_time']:.4f}毫秒/样本")
        print(f"模型大小: {model_size:.2f} KB")

        return results

    except Exception as e:
        print(f"✗ 加载Keras模型失败: {e}")
        return None


def test_tflite_model(tflite_path, x_test, y_test):
    """测试TFLite模型"""
    model_name = os.path.basename(tflite_path)
    print("\n" + "="*70)
    print(f"测试TFLite模型: {model_name}")
    print("="*70)
    
    try:
        # 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        
        # 获取输入输出详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.allocate_tensors()
        
        print(f"✓ 成功加载模型: {tflite_path}")
        print(f"输入类型: {input_details[0]['dtype']}")
        print(f"输出类型: {output_details[0]['dtype']}")
        
        # 获取量化参数
        input_quant = input_details[0].get('quantization_parameters', {})
        output_quant = output_details[0].get('quantization_parameters', {})
        
        input_scale = input_quant.get('scales', np.array([0.0]))
        input_zero_point = input_quant.get('zero_points', np.array([0]))
        output_scale = output_quant.get('scales', np.array([0.0]))
        output_zero_point = output_quant.get('zero_points', np.array([0]))
        
        is_input_quantized = input_details[0]['dtype'] == np.int8
        is_output_quantized = output_details[0]['dtype'] == np.int8
        
        if is_input_quantized:
            print(f"输入量化参数: scale={input_scale}, zero_point={input_zero_point}")
        if is_output_quantized:
            print(f"输出量化参数: scale={output_scale}, zero_point={output_zero_point}")
        
        # 预测
        predictions = []
        start_time = time.time()
        
        for i in range(len(x_test)):
            # 准备输入数据
            input_data = x_test[i:i+1].astype(np.float32)
            
            if is_input_quantized and len(input_scale) > 0 and input_scale[0] > 0:
                # 量化输入
                input_data = input_data / input_scale[0] + input_zero_point[0]
                input_data = np.clip(input_data, -128, 127).astype(np.int8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            if is_output_quantized and len(output_scale) > 0 and output_scale[0] > 0:
                # 反量化输出
                output_data = (output_data.astype(np.float32) - output_zero_point[0]) * output_scale[0]
            
            predictions.append(output_data[0])
        
        inference_time = time.time() - start_time
        predictions = np.array(predictions)
        y_pred = np.argmax(predictions, axis=1)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # 获取模型大小
        model_size = os.path.getsize(tflite_path) / 1024  # KB
        
        results = {
            'model_name': model_name.replace('ecgformer_', '').replace('.tflite', '').upper(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'avg_inference_time': inference_time / len(x_test) * 1000,  # ms per sample
            'model_size_mb': model_size / 1024,  # Convert to MB
            'model_size_kb': model_size,
            'predictions': y_pred,
            'is_quantized': is_input_quantized
        }
        
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"总推理时间: {inference_time:.2f}秒")
        print(f"平均推理时间: {results['avg_inference_time']:.4f}毫秒/样本")
        print(f"模型大小: {model_size:.2f} KB")
        
        return results
    
    except Exception as e:
        print(f"✗ 测试TFLite模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_comparison_charts(all_results, y_test, output_dir):
    """绘制对比图表"""
    print("\n" + "="*70)
    print("绘制对比图表")
    print("="*70)

    # 准备数据
    model_names = [r['model_name'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    inference_times = [r['avg_inference_time'] for r in all_results]
    model_sizes = [r['model_size_kb'] for r in all_results]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 准确率对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 准确率对比
    ax1 = axes[0, 0]
    colors = ['#2E86AB' if i == 0 else '#A23B72' for i in range(len(model_names))]
    bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylim([min(accuracies) - 0.02, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45)

    # 在柱状图上添加数值
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=16)

    # F1分数对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_names, f1_scores, color=colors, alpha=0.8)
    ax2.set_ylabel('F1 Score', fontsize=16, fontweight='bold')
    ax2.set_title('Model F1 Score Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylim([min(f1_scores) - 0.02, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45)

    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.4f}',
                ha='center', va='bottom', fontsize=16)

    # 推理时间对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(model_names, inference_times, color=colors, alpha=0.8)
    ax3.set_ylabel('Avg Inference Time (ms/sample)', fontsize=16, fontweight='bold')
    ax3.set_title('Inference Speed Comparison', fontsize=16, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars3, inference_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}',
                ha='center', va='bottom', fontsize=16)

    # 模型大小对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(model_names, model_sizes, color=colors, alpha=0.8)
    ax4.set_ylabel('Model Size (KB)', fontsize=16, fontweight='bold')
    ax4.set_title('Model Size Comparison', fontsize=16, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.tick_params(axis='x', rotation=45)

    for bar, size in zip(bars4, model_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f}',
                ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'quantization_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存对比图表: {comparison_path}")
    plt.close()

    # 2. 综合对比雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 归一化指标 (越高越好)
    categories = ['Accuracy', 'F1 Score', 'Speed\n(Inverse)', 'Compression\n(Inverse)']

    # 计算归一化值
    max_time = max(inference_times)
    max_size = max(model_sizes)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # 为每个模型绘制雷达图
    colors_radar = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    for idx, result in enumerate(all_results[:5]):  # 最多显示5个模型
        values = [
            result['accuracy'],
            result['f1_score'],
            1 - (result['avg_inference_time'] / max_time),  # 速度越快越好
            1 - (result['model_size_mb'] / max_size)  # 大小越小越好
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'],
                color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Model Comparison\n(Normalized Metrics)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, linestyle='--', alpha=0.5)

    radar_path = os.path.join(output_dir, 'radar_comparison.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存雷达图: {radar_path}")
    plt.close()

    # 3. 混淆矩阵对比（选择Keras和最佳量化模型）
    if len(all_results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Keras模型混淆矩阵
        cm_keras = confusion_matrix(y_test, all_results[0]['predictions'])
        sns.heatmap(cm_keras, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Confusion Matrix - {all_results[0]["model_name"]}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=11)
        axes[0].set_ylabel('True Label', fontsize=11)

        # 最佳量化模型混淆矩阵（准确率最高的量化模型）
        best_quantized_idx = 1
        for i in range(1, len(all_results)):
            if all_results[i]['accuracy'] > all_results[best_quantized_idx]['accuracy']:
                best_quantized_idx = i

        cm_quant = confusion_matrix(y_test, all_results[best_quantized_idx]['predictions'])
        sns.heatmap(cm_quant, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title(f'Confusion Matrix - {all_results[best_quantized_idx]["model_name"]}',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        axes[1].set_ylabel('True Label', fontsize=11)

        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrices.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存混淆矩阵: {cm_path}")
        plt.close()


def save_detailed_report(all_results, output_dir):
    """保存详细报告"""
    report_path = os.path.join(output_dir, 'quantization_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TFLite模型量化效果评估报告\n")
        f.write("="*70 + "\n\n")

        # 基准模型（Keras）
        keras_result = all_results[0]
        f.write(f"基准模型 (Keras):\n")
        f.write(f"  准确率: {keras_result['accuracy']:.6f}\n")
        f.write(f"  F1分数: {keras_result['f1_score']:.6f}\n")
        f.write(f"  推理时间: {keras_result['avg_inference_time']:.4f} ms/sample\n")
        f.write(f"  模型大小: {keras_result['model_size_mb']:.2f} MB\n\n")

        f.write("-"*70 + "\n")
        f.write("量化模型对比:\n")
        f.write("-"*70 + "\n\n")

        for result in all_results[1:]:
            f.write(f"\n{result['model_name']}:\n")
            f.write(f"  准确率: {result['accuracy']:.6f} ")
            acc_diff = (result['accuracy'] - keras_result['accuracy']) * 100
            f.write(f"({acc_diff:+.2f}%)\n")

            f.write(f"  F1分数: {result['f1_score']:.6f} ")
            f1_diff = (result['f1_score'] - keras_result['f1_score']) * 100
            f.write(f"({f1_diff:+.2f}%)\n")

            f.write(f"  推理时间: {result['avg_inference_time']:.4f} ms/sample ")
            time_speedup = keras_result['avg_inference_time'] / result['avg_inference_time']
            f.write(f"({time_speedup:.2f}x)\n")

            f.write(f"  模型大小: {result['model_size_kb']:.2f} KB ")
            size_ratio = keras_result['model_size_mb'] * 1024 / result['model_size_kb']
            f.write(f"(压缩 {size_ratio:.2f}x)\n")

        f.write("\n" + "="*70 + "\n")
        f.write("总结:\n")
        f.write("="*70 + "\n")

        # 找出最佳模型
        best_acc_idx = max(range(1, len(all_results)), key=lambda i: all_results[i]['accuracy'])
        best_speed_idx = min(range(1, len(all_results)), key=lambda i: all_results[i]['avg_inference_time'])
        best_size_idx = min(range(1, len(all_results)), key=lambda i: all_results[i]['model_size_mb'])

        f.write(f"\n最佳准确率量化模型: {all_results[best_acc_idx]['model_name']} ")
        f.write(f"(准确率: {all_results[best_acc_idx]['accuracy']:.6f})\n")

        f.write(f"最快推理速度模型: {all_results[best_speed_idx]['model_name']} ")
        f.write(f"(推理时间: {all_results[best_speed_idx]['avg_inference_time']:.4f} ms/sample)\n")

        f.write(f"最小模型大小: {all_results[best_size_idx]['model_name']} ")
        f.write(f"(大小: {all_results[best_size_idx]['model_size_kb']:.2f} KB)\n")

    print(f"✓ 保存详细报告: {report_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("TFLite模型量化效果测试")
    print("="*70)

    # 加载测试数据
    x_test, y_test = load_test_data()

    all_results = []

    # 测试原始Keras模型
    if os.path.exists(KERAS_MODEL_PATH):
        keras_result = test_keras_model(KERAS_MODEL_PATH, x_test, y_test)
        if keras_result:
            all_results.append(keras_result)
    else:
        print(f"警告: 未找到Keras模型 {KERAS_MODEL_PATH}")

    # 测试所有TFLite模型
    tflite_models = [
        'ecgformer_ptq_dynamic.tflite',
        'ecgformer_ptq_float16.tflite',
        #'ecgformer_ptq_int8.tflite',
        #'ecgformer_qat_int8.tflite'
    ]

    for tflite_model in tflite_models:
        tflite_path = os.path.join(TFLITE_DIR, tflite_model)
        if os.path.exists(tflite_path):
            result = test_tflite_model(tflite_path, x_test, y_test)
            if result:
                all_results.append(result)
        else:
            print(f"警告: 未找到TFLite模型 {tflite_path}")

    # 生成对比图表
    if len(all_results) > 0:
        plot_comparison_charts(all_results, y_test, OUTPUT_DIR)
        save_detailed_report(all_results, OUTPUT_DIR)

        # 打印摘要
        print("\n" + "="*70)
        print("测试完成！摘要:")
        print("="*70)
        print(f"\n{'模型':<25} {'准确率':<12} {'F1分数':<12} {'推理时间(ms)':<15} {'大小(MB)':<10}")
        print("-"*70)
        for result in all_results:
            print(f"{result['model_name']:<25} {result['accuracy']:<12.6f} {result['f1_score']:<12.6f} "
                  f"{result['avg_inference_time']:<15.4f} {result['model_size_mb']:<10.2f}")

        print(f"\n结果保存在: {OUTPUT_DIR}/")
        print("  - quantization_comparison.png: 四维对比图")
        print("  - radar_comparison.png: 雷达图综合对比")
        print("  - confusion_matrices.png: 混淆矩阵对比")
        print("  - quantization_report.txt: 详细文本报告")
    else:
        print("\n错误: 没有成功测试任何模型！")


if __name__ == '__main__':
    main()
