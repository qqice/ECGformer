"""
使用原始Keras模型进行ECG信号推理
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import argparse

# 类别映射
CLASS_NAMES = {
    0: 'N (Normal beat)',
    1: 'S (Supraventricular premature beat)',
    2: 'V (Premature ventricular contraction)',
    3: 'F (Fusion of ventricular and normal beat)',
    4: 'Q (Unclassifiable beat)'
}

def load_model(model_path='./tmp/best_model.keras'):
    """加载训练好的Keras模型"""
    print(f"正在加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    print("模型加载成功！")
    return model

def preprocess_signal(signal):
    """
    预处理ECG信号

    参数:
        signal: 长度为187的ECG信号数组

    返回:
        预处理后的信号，形状为 (1, 187, 1)
    """
    signal = np.array(signal).reshape(1, -1)

    # 标准化
    scaler = StandardScaler()
    signal = scaler.fit_transform(signal)

    # 调整形状为模型输入格式
    signal = signal.reshape((1, signal.shape[1], 1))

    return signal

def predict_signal(model, signal):
    """
    预测ECG信号类型

    参数:
        model: 加载的Keras模型
        signal: 预处理后的信号

    返回:
        predicted_class: 预测的类别ID
        probabilities: 各类别的概率
    """
    probabilities = model.predict(signal, verbose=0)[0]
    predicted_class = np.argmax(probabilities)

    return predicted_class, probabilities

def inference_from_csv(model, csv_path, sample_idx=0):
    """
    从CSV文件中读取一个样本并进行推理

    参数:
        model: 加载的模型
        csv_path: CSV文件路径
        sample_idx: 要推理的样本索引
    """
    print(f"\n从CSV文件读取样本: {csv_path}, 索引: {sample_idx}")

    # 读取CSV
    data = pd.read_csv(csv_path, header=None)

    if sample_idx >= len(data):
        print(f"错误: 样本索引 {sample_idx} 超出范围（总样本数: {len(data)}）")
        return

    # 提取特征和真实标签
    features = data.iloc[sample_idx, :-1].values
    true_label = int(data.iloc[sample_idx, -1])

    print(f"信号长度: {len(features)}")
    print(f"真实类别: {true_label} - {CLASS_NAMES[true_label]}")

    # 预处理
    processed_signal = preprocess_signal(features)

    # 推理
    predicted_class, probabilities = predict_signal(model, processed_signal)

    # 显示结果
    print("\n" + "="*60)
    print("推理结果:")
    print("="*60)
    print(f"预测类别: {predicted_class} - {CLASS_NAMES[predicted_class]}")
    print(f"真实类别: {true_label} - {CLASS_NAMES[true_label]}")
    print(f"预测正确: {'✓' if predicted_class == true_label else '✗'}")

    print("\n各类别预测概率:")
    for i, prob in enumerate(probabilities):
        print(f"  类别 {i} ({CLASS_NAMES[i]}): {prob:.4f} ({prob*100:.2f}%)")

    return predicted_class, probabilities, true_label

def inference_from_array(model, signal_array):
    """
    直接从数组进行推理

    参数:
        model: 加载的模型
        signal_array: 长度为187的ECG信号数组
    """
    print(f"\n从数组进行推理，信号长度: {len(signal_array)}")

    # 预处理
    processed_signal = preprocess_signal(signal_array)

    # 推理
    predicted_class, probabilities = predict_signal(model, processed_signal)

    # 显示结果
    print("\n" + "="*60)
    print("推理结果:")
    print("="*60)
    print(f"预测类别: {predicted_class} - {CLASS_NAMES[predicted_class]}")

    print("\n各类别预测概率:")
    for i, prob in enumerate(probabilities):
        print(f"  类别 {i} ({CLASS_NAMES[i]}): {prob:.4f} ({prob*100:.2f}%)")

    return predicted_class, probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECG信号推理（使用Keras模型）')
    parser.add_argument('--model', type=str, default='./tmp/best_model.keras',
                        help='模型路径')
    parser.add_argument('--csv', type=str, default='mitbih_test.csv',
                        help='CSV数据文件路径')
    parser.add_argument('--index', type=int, default=0,
                        help='要推理的样本索引')

    args = parser.parse_args()

    # 加载模型
    model = load_model(args.model)

    # 从CSV推理
    inference_from_csv(model, args.csv, args.index)

    print("\n" + "="*60)
    print("示例: 推理多个样本")
    print("="*60)
    for idx in [1, 10, 100]:
        print(f"\n>>> 样本索引 {idx}:")
        pred, probs, true = inference_from_csv(model, args.csv, idx)

