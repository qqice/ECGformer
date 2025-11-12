"""
简化版模型层输出对比工具 - 修复TFLite层匹配问题

使用方法:
    python compare_model_layers_v2.py --tflite_model ./exported_models/tflite/ecgformer_ptq_dynamic.tflite --num_samples 20
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# 配置
MODEL_PATH = './ckpts/best_model.keras'
TEST_DATA_PATH = './dataset/mitbih_test.csv'
OUTPUT_DIR = './results'

# 禁用GPU，使用CPU以避免初始化问题
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

def get_tflite_outputs(tflite_path, data):
    """提取TFLite模型各层输出"""
    print("\n提取TFLite模型各层输出...")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()
    
    print(f"TFLite模型共有 {len(tensor_details)} 个张量")
    
    # 识别可提取的激活张量
    print("\n识别激活张量...")
    activations = []
    for detail in tensor_details:
        name = detail['name']
        shape = detail['shape']
        idx = detail['index']
        
        # 过滤条件：
        # 1. 不是常量
        if 'constant' in name.lower() or 'pseudo' in name.lower():
            continue
        # 2. batch维度应该是1
        if len(shape) == 0 or shape[0] != 1:
            continue
        
        activations.append((idx, name, tuple(shape)))
        print(f"  {idx:3d}. {name:60s} | Shape: {shape}")
    
    print(f"\n找到 {len(activations)} 个激活张量")
    
    # 提取输出
    print(f"\n提取 {len(data)} 个样本的输出...")
    results = {}
    
    for act_idx, act_name, act_shape in activations:
        outputs = []
        
        for i in range(len(data)):
            if (i + 1) % 10 == 0:
                print(f"  样本 {i+1}/{len(data)}...")
            
            sample = data[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            
            try:
                tensor = interpreter.get_tensor(act_idx)
                outputs.append(tensor.copy())
            except:
                break
        
        if len(outputs) == len(data):
            stacked = np.concatenate(outputs, axis=0)
            results[act_name] = stacked
    
    print(f"\n成功提取 {len(results)} 个张量")
    return results

def match_layers(keras_outs, tflite_outs):
    """匹配Keras和TFLite的层"""
    print("\n" + "="*80)
    print("匹配Keras层和TFLite张量...")
    print("="*80)
    
    matches = []
    used_tflite = set()
    
    # 策略1: 通过名称关键词匹配
    print("\n【策略1】名称关键词匹配:")
    import re
    
    for k_name, k_out in keras_outs.items():
        k_shape = k_out.shape
        
        # 提取关键词和数字
        k_words = set(k_name.lower().split('_'))
        k_nums = set(re.findall(r'\d+', k_name))
        
        best_match = None
        best_score = 0
        
        for t_name, t_out in tflite_outs.items():
            if t_name in used_tflite or t_out.shape != k_shape:
                continue
            
            t_words = set(t_name.lower().split('_'))
            t_words.update(t_name.lower().split('/'))
            t_nums = set(re.findall(r'\d+', t_name))
            
            # 计算匹配分数
            score = 0
            # 关键词交集
            common_words = k_words & t_words
            score += len(common_words) * 10
            
            # 数字匹配
            if k_nums and t_nums and k_nums & t_nums:
                score += 20
            
            # 特殊关键词强匹配
            for keyword in ['normalization', 'attention', 'conv', 'dense', 'add', 'pooling']:
                if keyword in k_name.lower() and keyword in t_name.lower():
                    score += 15
            
            if score > best_score:
                best_score = score
                best_match = (t_name, t_out)
        
        if best_match and best_score >= 10:
            matches.append((k_name, best_match[0], k_out, best_match[1]))
            used_tflite.add(best_match[0])
            print(f"  ✓ [{best_score:3d}分] {k_name:35s} <-> {best_match[0][:60]}")
    
    # 策略2: 形状+位置顺序匹配
    print(f"\n【策略2】形状顺序匹配:")
    remaining_k = [(n, o) for n, o in keras_outs.items() if not any(n == m[0] for m in matches)]
    remaining_t = [(n, o) for n, o in tflite_outs.items() if n not in used_tflite]
    
    # 按形状分组
    from collections import defaultdict
    k_by_shape = defaultdict(list)
    t_by_shape = defaultdict(list)
    
    for n, o in remaining_k:
        k_by_shape[o.shape].append((n, o))
    for n, o in remaining_t:
        t_by_shape[o.shape].append((n, o))
    
    for shape in k_by_shape:
        if shape in t_by_shape:
            k_list = k_by_shape[shape]
            t_list = t_by_shape[shape]
            
            for i, (k_n, k_o) in enumerate(k_list):
                if i < len(t_list):
                    t_n, t_o = t_list[i]
                    matches.append((k_n, t_n, k_o, t_o))
                    used_tflite.add(t_n)
                    print(f"  ✓ {k_n:35s} <-> {t_n[:60]}")
    
    print(f"\n总共匹配 {len(matches)} 对层")
    return matches

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
    parser = argparse.ArgumentParser(description='对比Keras和TFLite模型层输出')
    parser.add_argument('--keras_model', type=str, default=MODEL_PATH)
    parser.add_argument('--tflite_model', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()
    
    print("="*80)
    print("模型层输出对比工具 v2")
    print("="*80)
    print(f"Keras模型: {args.keras_model}")
    print(f"TFLite模型: {args.tflite_model}")
    print(f"样本数: {args.num_samples}")
    
    # 加载数据
    print("\n加载数据...")
    x_test, y_test = readucr(TEST_DATA_PATH)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_test = x_test[:args.num_samples]
    print(f"数据形状: {x_test.shape}")
    
    # 加载模型
    print("\n加载Keras模型...")
    keras_model = tf.keras.models.load_model(args.keras_model)
    
    # 提取输出
    keras_outs = get_keras_outputs(keras_model, x_test)
    tflite_outs = get_tflite_outputs(args.tflite_model, x_test)
    
    # 匹配层
    matches = match_layers(keras_outs, tflite_outs)
    
    if not matches:
        print("\n⚠️ 未找到匹配的层！")
        return
    
    # 计算误差
    print("\n" + "="*80)
    print("计算误差指标...")
    print("="*80)
    
    results = []
    for k_name, t_name, k_out, t_out in matches:
        metrics = calculate_metrics(k_out, t_out)
        results.append({
            'Keras Layer': k_name,
            'TFLite Tensor': t_name[:50],  # 截断长名称
            'Shape': str(k_out.shape),
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
    csv_path = os.path.join(OUTPUT_DIR, f'layer_comparison_{basename}_v2.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 结果已保存到: {csv_path}")
    
    # 统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"匹配层数: {len(df)}")
    print(f"平均MAE: {df['MAE'].mean():.6f}")
    print(f"平均RMSE: {df['RMSE'].mean():.6f}")
    print(f"平均MAPE: {df['MAPE(%)'].mean():.2f}%")
    print(f"平均SNR: {df['SNR(dB)'].mean():.2f} dB")
    print(f"平均余弦相似度: {df['Cosine Sim'].mean():.6f}")

if __name__ == '__main__':
    main()

