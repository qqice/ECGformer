"""
评估ONNX模型在测试集上的效果，并与原始Keras模型对比
使用批处理方式避免内存溢出
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import onnxruntime as ort
from tensorflow import keras
import time
import argparse
import gc

# 类别映射
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_FULL_NAMES = {
    0: 'N (Normal beat)',
    1: 'S (Supraventricular premature beat)',
    2: 'V (Premature ventricular contraction)',
    3: 'F (Fusion of ventricular and normal beat)',
    4: 'Q (Unclassifiable beat)'
}

def load_test_data_generator(csv_path='mitbih_test.csv', batch_size=512):
    """
    加载测试数据，预先计算标准化参数以避免内存问题
    """
    print(f"正在加载测试数据: {csv_path}")

    # 读取数据
    data = pd.read_csv(csv_path, header=None)
    y_all = data.iloc[:, -1].astype(int).values

    total_samples = len(data)
    print(f"测试集大小: {total_samples} 样本")
    print(f"类别分布: {np.bincount(y_all)}")

    # 预先计算标准化参数（只用统计量，不保存转换后的数据）
    print("计算标准化参数...")
    X_all = data.iloc[:, :-1].values
    scaler = StandardScaler()
    scaler.fit(X_all)

    # 释放完整数据
    del X_all
    gc.collect()

    return data, y_all, scaler, total_samples

def evaluate_keras_model_batched(model_path, data, y_test, scaler, batch_size=512):
    """评估Keras模型（批处理版本）"""
    print("\n" + "="*60)
    print("评估Keras原始模型")
    print("="*60)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = keras.models.load_model(model_path)

    total_samples = len(data)
    y_pred_all = []

    # 推理时间测试
    start_time = time.time()

    # 批处理推理
    print(f"开始批处理推理 (批大小: {batch_size})...")
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)

        # 读取批次数据
        X_batch = data.iloc[start_idx:end_idx, :-1].values

        # 使用预计算的标准化参数
        X_batch = scaler.transform(X_batch)

        # 调整形状
        X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], 1))

        # 推理
        y_pred_probs_batch = model.predict(X_batch, verbose=0)
        y_pred_batch = np.argmax(y_pred_probs_batch, axis=1)

        y_pred_all.extend(y_pred_batch)

        # 显示进度
        progress = (end_idx / total_samples) * 100
        print(f"\r进度: {end_idx}/{total_samples} ({progress:.1f}%)", end='', flush=True)

        # 释放内存
        del X_batch, y_pred_probs_batch
        gc.collect()

    inference_time = time.time() - start_time
    print()  # 换行

    y_pred = np.array(y_pred_all)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n推理时间: {inference_time:.4f} 秒")
    print(f"平均每个样本: {inference_time/len(y_test)*1000:.4f} 毫秒")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return y_pred, inference_time, accuracy

def evaluate_onnx_model_batched(model_path, data, y_test, scaler, batch_size=512):
    """评估ONNX模型（批处理版本）"""
    print("\n" + "="*60)
    print("评估ONNX模型")
    print("="*60)

    # 加载ONNX模型
    print(f"加载ONNX模型: {model_path}")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total_samples = len(data)
    y_pred_all = []

    # 推理时间测试
    start_time = time.time()

    # 批处理推理
    print(f"开始批处理推理 (批大小: {batch_size})...")
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)

        # 读取批次数据
        X_batch = data.iloc[start_idx:end_idx, :-1].values

        # 使用预计算的标准化参数
        X_batch = scaler.transform(X_batch)

        # 调整形状并转换为float32
        X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], 1)).astype(np.float32)

        # 推理
        outputs = session.run(None, {input_name: X_batch})
        y_pred_batch = np.argmax(outputs[0], axis=1)

        y_pred_all.extend(y_pred_batch)

        # 显示进度
        progress = (end_idx / total_samples) * 100
        print(f"\r进度: {end_idx}/{total_samples} ({progress:.1f}%)", end='', flush=True)

        # 释放内存
        del X_batch, outputs
        gc.collect()

    inference_time = time.time() - start_time
    print()  # 换行

    y_pred = np.array(y_pred_all)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n推理时间: {inference_time:.4f} 秒")
    print(f"平均每个样本: {inference_time/len(y_test)*1000:.4f} 毫秒")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return y_pred, inference_time, accuracy

def compare_predictions(y_pred_keras, y_pred_onnx, y_test):
    """比较两个模型的预测结果"""
    print("\n" + "="*60)
    print("模型预测对比")
    print("="*60)

    # 预测一致性
    agreement = np.mean(y_pred_keras == y_pred_onnx)
    print(f"预测一致性: {agreement:.4f} ({agreement*100:.2f}%)")

    # 找出不一致的样本
    disagreement_indices = np.where(y_pred_keras != y_pred_onnx)[0]
    print(f"不一致样本数: {len(disagreement_indices)}")

    if len(disagreement_indices) > 0 and len(disagreement_indices) <= 10:
        print("\n不一致的样本:")
        for idx in disagreement_indices[:10]:
            print(f"  样本 {idx}: Keras预测={y_pred_keras[idx]}, "
                  f"ONNX预测={y_pred_onnx[idx]}, 真实={y_test[idx]}")

def plot_comparison(y_test, y_pred_keras, y_pred_onnx,
                    keras_time, onnx_time,
                    keras_acc, onnx_acc):
    """绘制对比图"""
    print("\n生成对比图表...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 混淆矩阵 - Keras
    cm_keras = confusion_matrix(y_test, y_pred_keras, labels=[0, 1, 2, 3, 4])
    disp_keras = ConfusionMatrixDisplay(confusion_matrix=cm_keras,
                                        display_labels=CLASS_NAMES)
    disp_keras.plot(ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title(f'Keras Model Confusion Matrix\nAccuracy: {keras_acc:.4f}')

    # 2. 混淆矩阵 - ONNX
    cm_onnx = confusion_matrix(y_test, y_pred_onnx, labels=[0, 1, 2, 3, 4])
    disp_onnx = ConfusionMatrixDisplay(confusion_matrix=cm_onnx,
                                       display_labels=CLASS_NAMES)
    disp_onnx.plot(ax=axes[0, 1], cmap='Greens')
    axes[0, 1].set_title(f'ONNX Model Confusion Matrix\nAccuracy: {onnx_acc:.4f}')

    # 3. 推理时间对比
    models = ['Keras', 'ONNX']
    times = [keras_time, onnx_time]
    colors = ['#3498db', '#2ecc71']

    bars = axes[1, 0].bar(models, times, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Inference Time (seconds)')
    axes[1, 0].set_title('Inference Time Comparison')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 在柱状图上添加数值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{time_val:.4f}s\n({time_val/len(y_test)*1000:.4f}ms/sample)',
                       ha='center', va='bottom')

    # 4. 准确率对比
    accuracies = [keras_acc * 100, onnx_acc * 100]
    bars = axes[1, 1].bar(models, accuracies, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy Comparison')
    axes[1, 1].set_ylim([min(accuracies) - 1, 100])
    axes[1, 1].grid(axis='y', alpha=0.3)

    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%',
                       ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('./results/model_comparison.jpg', dpi=300, bbox_inches='tight')
    print("对比图已保存为: ./results/model_comparison.jpg")
    plt.close()

def save_detailed_report(y_test, y_pred_keras, y_pred_onnx,
                        keras_time, onnx_time,
                        keras_acc, onnx_acc,
                        output_file='./results/evaluation_report.txt'):
    """保存详细评估报告"""
    print(f"\n保存详细报告到: {output_file}")

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ECG模型评估对比报告\n")
        f.write("="*60 + "\n\n")

        f.write("1. 整体性能对比\n")
        f.write("-"*60 + "\n")
        f.write(f"模型类型          准确率          推理时间        平均推理时间\n")
        f.write(f"Keras原始模型    {keras_acc:.4f}      {keras_time:.4f}s     {keras_time/len(y_test)*1000:.4f}ms\n")
        f.write(f"ONNX模型         {onnx_acc:.4f}      {onnx_time:.4f}s     {onnx_time/len(y_test)*1000:.4f}ms\n")
        f.write(f"\n速度提升: {keras_time/onnx_time:.2f}x\n")
        f.write(f"准确率差异: {abs(keras_acc - onnx_acc):.6f}\n\n")

        f.write("2. Keras模型详细分类报告\n")
        f.write("-"*60 + "\n")
        report_keras = classification_report(y_test, y_pred_keras,
                                            labels=[0, 1, 2, 3, 4],
                                            target_names=CLASS_NAMES,
                                            zero_division=0)
        f.write(report_keras + "\n\n")

        f.write("3. ONNX模型详细分类报告\n")
        f.write("-"*60 + "\n")
        report_onnx = classification_report(y_test, y_pred_onnx,
                                           labels=[0, 1, 2, 3, 4],
                                           target_names=CLASS_NAMES,
                                           zero_division=0)
        f.write(report_onnx + "\n\n")

        f.write("4. 预测一致性分析\n")
        f.write("-"*60 + "\n")
        agreement = np.mean(y_pred_keras == y_pred_onnx)
        f.write(f"预测一致性: {agreement:.4f} ({agreement*100:.2f}%)\n")
        disagreement_indices = np.where(y_pred_keras != y_pred_onnx)[0]
        f.write(f"不一致样本数: {len(disagreement_indices)}/{len(y_test)}\n\n")

        f.write("5. 结论\n")
        f.write("-"*60 + "\n")
        if abs(keras_acc - onnx_acc) < 0.001:
            f.write("✓ ONNX模型与原始Keras模型准确率基本一致\n")
        else:
            f.write("⚠ ONNX模型与原始Keras模型准确率存在差异\n")

        if onnx_time < keras_time:
            f.write(f"✓ ONNX模型推理速度更快 ({keras_time/onnx_time:.2f}x)\n")
        else:
            f.write("⚠ ONNX模型推理速度较慢\n")

        if agreement > 0.99:
            f.write("✓ 两个模型预测高度一致\n")
        else:
            f.write("⚠ 两个模型预测存在一定差异\n")

    print("报告保存成功！")

def main():
    parser = argparse.ArgumentParser(description='评估ONNX模型并与Keras模型对比')
    parser.add_argument('--keras_model', type=str, default='./ckpts/best_model.keras',
                        help='Keras模型路径')
    parser.add_argument('--onnx_model', type=str, default='./exported_models/ecgformer_model.onnx',
                        help='ONNX模型路径')
    parser.add_argument('--test_csv', type=str, default='mitbih_test.csv',
                        help='测试数据CSV路径')
    parser.add_argument('--output', type=str, default='evaluation_report.txt',
                        help='输出报告文件名')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='批处理大小（默认512，可根据内存调整）')

    args = parser.parse_args()

    print("="*60)
    print("ECG模型评估与对比 (批处理版本)")
    print("="*60)
    print(f"批大小: {args.batch_size}")
    print("提示: 如果仍然遇到内存问题，请使用 --batch_size 参数减小批大小")
    print("="*60)

    # 1. 加载测试数据
    data, y_test, scaler, total_samples = load_test_data_generator(args.test_csv, args.batch_size)

    # 2. 评估Keras模型
    y_pred_keras, keras_time, keras_acc = \
        evaluate_keras_model_batched(args.keras_model, data, y_test, scaler, args.batch_size)

    # 释放内存
    gc.collect()

    # 3. 评估ONNX模型
    y_pred_onnx, onnx_time, onnx_acc = \
        evaluate_onnx_model_batched(args.onnx_model, data, y_test, scaler, args.batch_size)

    # 释放内存
    gc.collect()

    # 4. 对比预测结果
    compare_predictions(y_pred_keras, y_pred_onnx, y_test)

    # 5. 生成对比图表
    plot_comparison(y_test, y_pred_keras, y_pred_onnx,
                   keras_time, onnx_time,
                   keras_acc, onnx_acc)

    # 6. 保存详细报告
    save_detailed_report(y_test, y_pred_keras, y_pred_onnx,
                        keras_time, onnx_time,
                        keras_acc, onnx_acc,
                        args.output)

    # 7. 总结
    print("\n" + "="*60)
    print("评估完成总结")
    print("="*60)
    print(f"Keras模型准确率: {keras_acc:.4f} ({keras_acc*100:.2f}%)")
    print(f"ONNX模型准确率:  {onnx_acc:.4f} ({onnx_acc*100:.2f}%)")
    print(f"准确率差异:      {abs(keras_acc - onnx_acc):.6f}")
    print(f"\nKeras推理时间:   {keras_time:.4f}秒")
    print(f"ONNX推理时间:    {onnx_time:.4f}秒")
    print(f"速度提升:        {keras_time/onnx_time:.2f}x")
    print(f"\n详细报告已保存至: {args.output}")
    print(f"对比图表已保存至: model_comparison.jpg")

if __name__ == "__main__":
    main()
