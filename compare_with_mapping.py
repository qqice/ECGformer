"""
精确层对比工具 - 基于手动构建的正确映射表

使用预先定义的正确映射关系，精确对比 Keras 和 TFLite 模型的每一层输出

使用方法:
    python compare_with_mapping.py --num_samples 20
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from layer_mapping import LAYER_MAPPING, is_quantized

# 配置
MODEL_PATH = './ckpts/best_model.keras'
TFLITE_MODEL_PATH = './exported_models/tflite/ecgformer_ptq_dynamic.tflite'
TEST_DATA_PATH = './dataset/mitbih_test.csv'
OUTPUT_DIR = './results'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def readucr(filename):
    """读取并预处理数据"""
    data = pd.read_csv(filename, header=None)
    y = data.iloc[:, -1].astype(int).to_numpy()
    x = data.iloc[:, :-1]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x)
    x = pd.DataFrame(standardized_data, columns=x.columns).to_numpy()
    return x, y.astype(int)

def get_keras_outputs(model, data):
    """提取Keras模型各层输出"""
    print("\n提取Keras模型各层输出...")
    outputs = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        submodel = tf.keras.Model(inputs=model.input, outputs=layer.output)
        out = submodel.predict(data, verbose=0)
        outputs[layer.name] = out
        print(f"  {i:2d}. {layer.name:40s} | Shape: {out.shape}")
    return outputs

def get_tflite_outputs(tflite_path, data, target_tensor_names):
    """提取TFLite模型指定张量的输出"""
    print("\n提取TFLite模型张量输出...")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 建立名称到索引的映射（包括中间张量和最终输出）
    name_to_idx = {}
    for detail in tensor_details:
        name_to_idx[detail['name']] = detail['index']
    # 添加最终输出张量
    for detail in output_details:
        name_to_idx[detail['name']] = detail['index']
    
    # 提取目标张量
    results = {}
    found_count = 0
    
    for target_name in target_tensor_names:
        if target_name not in name_to_idx:
            print(f"  ⚠️  未找到张量: {target_name}")
            continue
        
        idx = name_to_idx[target_name]
        outputs = []
        
        for i in range(len(data)):
            sample = data[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            
            try:
                tensor_data = interpreter.get_tensor(idx)
                outputs.append(tensor_data.copy())
            except Exception as e:
                print(f"  ✗ 提取失败: {target_name} - {e}")
                break
        
        if len(outputs) == len(data):
            stacked = np.concatenate(outputs, axis=0)
            results[target_name] = stacked
            found_count += 1
    
    print(f"✓ 成功提取 {found_count}/{len(target_tensor_names)} 个目标张量")
    return results

def calculate_metrics(orig, quant):
    """计算误差指标"""
    o_flat = orig.flatten()
    q_flat = quant.flatten()
    
    mae = np.mean(np.abs(o_flat - q_flat))
    rmse = np.sqrt(np.mean((o_flat - q_flat) ** 2))
    
    # MAPE
    mask = np.abs(o_flat) > 1e-10
    mape = np.mean(np.abs((o_flat[mask] - q_flat[mask]) / o_flat[mask])) * 100 if mask.sum() > 0 else 0
    
    # SNR
    signal_power = np.mean(o_flat ** 2)
    noise_power = np.mean((o_flat - q_flat) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-10 else float('inf')
    
    # Cosine similarity
    norm_o = np.linalg.norm(o_flat)
    norm_q = np.linalg.norm(q_flat)
    cos_sim = np.dot(o_flat, q_flat) / (norm_o * norm_q) if norm_o > 1e-10 and norm_q > 1e-10 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE(%)': mape,
        'Max Error': np.max(np.abs(o_flat - q_flat)),
        'SNR(dB)': snr,
        'Cosine Sim': cos_sim
    }

def main():
    parser = argparse.ArgumentParser(description='精确对比Keras和TFLite模型层输出（基于正确映射）')
    parser.add_argument('--keras_model', type=str, default=MODEL_PATH)
    parser.add_argument('--tflite_model', type=str, default=TFLITE_MODEL_PATH)
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()
    
    print("="*80)
    print("精确层输出对比工具（基于手动映射表）")
    print("="*80)
    print(f"Keras模型: {args.keras_model}")
    print(f"TFLite模型: {args.tflite_model}")
    print(f"样本数: {args.num_samples}")
    print(f"映射表: layer_mapping.py ({len(LAYER_MAPPING)} 个映射)")
    
    # 加载数据
    print("\n加载数据...")
    x_test, y_test = readucr(TEST_DATA_PATH)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_test = x_test[:args.num_samples]
    print(f"数据形状: {x_test.shape}")
    
    # 加载Keras模型
    print("\n加载Keras模型...")
    keras_model = tf.keras.models.load_model(args.keras_model)
    
    # 提取Keras输出
    keras_outs = get_keras_outputs(keras_model, x_test)
    
    # 获取需要提取的TFLite张量名称
    target_tensors = set(LAYER_MAPPING.values())
    
    # 提取TFLite输出
    tflite_outs = get_tflite_outputs(args.tflite_model, x_test, target_tensors)
    
    # 计算误差
    print("\n" + "="*80)
    print("计算误差指标...")
    print("="*80)
    
    results = []
    for keras_name, tflite_name in LAYER_MAPPING.items():
        if keras_name not in keras_outs:
            print(f"⚠️  Keras层 '{keras_name}' 不存在")
            continue
        
        if tflite_name not in tflite_outs:
            print(f"⚠️  TFLite张量 '{tflite_name}' 未提取")
            continue
        
        keras_out = keras_outs[keras_name]
        tflite_out = tflite_outs[tflite_name]
        
        # 检查形状是否匹配
        if keras_out.shape != tflite_out.shape:
            print(f"⚠️  形状不匹配: {keras_name} {keras_out.shape} vs {tflite_out.shape}")
            continue
        
        metrics = calculate_metrics(keras_out, tflite_out)
        
        results.append({
            'Keras Layer': keras_name,
            'TFLite Tensor': tflite_name[:60],
            'Shape': str(keras_out.shape),
            'Quantized': '✓' if is_quantized(keras_name) else '',
            **metrics
        })
    
    df = pd.DataFrame(results)
    
    # 显示结果
    print("\n" + "="*80)
    print("层输出误差对比表")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.6f'))
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.tflite_model))[0]
    csv_path = os.path.join(OUTPUT_DIR, f'layer_comparison_{basename}_accurate.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 结果已保存到: {csv_path}")
    
    # 统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"总层数: {len(df)}")
    print(f"平均MAE: {df['MAE'].mean():.8f}")
    print(f"平均RMSE: {df['RMSE'].mean():.8f}")
    print(f"平均MAPE: {df['MAPE(%)'].mean():.4f}%")
    
    valid_snr = df[df['SNR(dB)'] != float('inf')]['SNR(dB)']
    if len(valid_snr) > 0:
        print(f"平均SNR: {valid_snr.mean():.2f} dB (排除inf)")
    
    print(f"平均余弦相似度: {df['Cosine Sim'].mean():.8f}")
    
    # 量化层分析
    print("\n" + "="*80)
    print("量化层分析")
    print("="*80)
    
    quantized_df = df[df['Quantized'] == '✓']
    non_quantized_df = df[df['Quantized'] == '']
    
    if len(quantized_df) > 0:
        print(f"\n量化层 ({len(quantized_df)} 层):")
        print(f"  平均MAE: {quantized_df['MAE'].mean():.8f}")
        print(f"  平均余弦相似度: {quantized_df['Cosine Sim'].mean():.8f}")
        print("\n  详细:")
        for _, row in quantized_df.iterrows():
            print(f"    • {row['Keras Layer']:30s} | MAE: {row['MAE']:.8f} | Cos: {row['Cosine Sim']:.8f}")
    
    if len(non_quantized_df) > 0:
        print(f"\n非量化层 ({len(non_quantized_df)} 层):")
        print(f"  平均MAE: {non_quantized_df['MAE'].mean():.8f}")
        print(f"  平均余弦相似度: {non_quantized_df['Cosine Sim'].mean():.8f}")
    
    # 误差最大的层
    print("\n" + "="*80)
    print("误差最大的5个层")
    print("="*80)
    df_sorted = df.sort_values('MAE', ascending=False)
    for idx, row in df_sorted.head(5).iterrows():
        quant_mark = '[量化]' if row['Quantized'] == '✓' else ''
        print(f"  {quant_mark:8s} {row['Keras Layer']:30s} | MAE: {row['MAE']:.8f} | Cos: {row['Cosine Sim']:.8f}")

if __name__ == '__main__':
    main()
