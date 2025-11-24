"""
构建 Keras 到 TFLite 的精确层映射表

这个脚本通过分析实际的输出数值来建立正确的映射关系
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_PATH = './ckpts/best_model.keras'
TFLITE_MODEL_PATH = './exported_models/tflite/ecgformer_ptq_dynamic.tflite'
TEST_DATA_PATH = './dataset/mitbih_test.csv'

def readucr(filename):
    data = pd.read_csv(filename, header=None)
    y = data.iloc[:, -1].astype(int).to_numpy()
    x = data.iloc[:, :-1]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x)
    x = pd.DataFrame(standardized_data, columns=x.columns).to_numpy()
    return x, y.astype(int)

def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return np.dot(a_flat, b_flat) / (norm_a * norm_b)

print("="*80)
print("构建精确的 Keras → TFLite 层映射表")
print("="*80)

# 加载数据
print("\n加载测试数据...")
x_test, y_test = readucr(TEST_DATA_PATH)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
x_test = x_test[:5]  # 使用5个样本
print(f"数据形状: {x_test.shape}")

# 加载 Keras 模型
print("\n加载 Keras 模型...")
keras_model = tf.keras.models.load_model(MODEL_PATH)

# 提取 Keras 各层输出
print("\n提取 Keras 各层输出...")
keras_outputs = {}
for i, layer in enumerate(keras_model.layers):
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    submodel = tf.keras.Model(inputs=keras_model.input, outputs=layer.output)
    out = submodel.predict(x_test, verbose=0)
    keras_outputs[layer.name] = out
    print(f"  {i:2d}. {layer.name:40s} | {out.shape}")

# 加载 TFLite 模型
print("\n加载 TFLite 模型...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()

# 提取所有 TFLite 张量
print("\n提取 TFLite 张量...")
tflite_tensors = {}

for detail in tensor_details:
    name = detail['name']
    shape = detail['shape']
    idx = detail['index']
    
    # 只处理批次维度为1的张量
    if len(shape) == 0 or shape[0] != 1:
        continue
    
    # 跳过常量
    if 'constant' in name.lower() or 'pseudo' in name.lower():
        continue
    
    # 提取该张量在所有样本上的输出
    outputs = []
    for i in range(len(x_test)):
        sample = x_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        
        try:
            tensor_data = interpreter.get_tensor(idx)
            outputs.append(tensor_data.copy())
        except:
            break
    
    if len(outputs) == len(x_test):
        stacked = np.concatenate(outputs, axis=0)
        tflite_tensors[name] = stacked

print(f"成功提取 {len(tflite_tensors)} 个 TFLite 张量")

# 建立映射
print("\n"+"="*80)
print("基于数值相似度建立映射表")
print("="*80)

mapping = {}
used_tflite = set()

for keras_name, keras_out in keras_outputs.items():
    keras_shape = keras_out.shape
    
    # 查找所有相同形状的 TFLite 张量
    candidates = []
    for tflite_name, tflite_out in tflite_tensors.items():
        if tflite_name in used_tflite:
            continue
        if tflite_out.shape != keras_shape:
            continue
        
        sim = cosine_similarity(keras_out, tflite_out)
        mae = np.mean(np.abs(keras_out - tflite_out))
        candidates.append((tflite_name, sim, mae))
    
    if not candidates:
        print(f"\n⚠️  {keras_name}: 未找到匹配")
        continue
    
    # 按相似度排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 显示最佳匹配
    best_name, best_sim, best_mae = candidates[0]
    
    print(f"\n{keras_name} → {best_name[:70]}")
    print(f"  相似度: {best_sim:.8f}, MAE: {best_mae:.8f}")
    
    # 显示其他候选（如果相似度也很高）
    if len(candidates) > 1:
        for i, (name, sim, mae) in enumerate(candidates[1:4], 1):
            if sim > 0.95:
                print(f"  候选{i}: {name[:60]} (sim={sim:.6f}, mae={mae:.6f})")
    
    # 只有相似度 > 0.95 才认为是正确匹配
    if best_sim > 0.95:
        mapping[keras_name] = best_name
        used_tflite.add(best_name)
        print(f"  ✓ 已确认映射")
    else:
        print(f"  ✗ 相似度太低，跳过")

# 保存映射表
print("\n"+"="*80)
print("保存映射表")
print("="*80)

# 生成 Python 代码格式的映射表
mapping_code = "# Keras 层名称 → TFLite 张量名称的精确映射\n"
mapping_code += "# 通过数值相似度验证的准确映射关系\n"
mapping_code += "LAYER_MAPPING = {\n"

for keras_name, tflite_name in mapping.items():
    mapping_code += f"    '{keras_name}': '{tflite_name}',\n"

mapping_code += "}\n"

with open('./layer_mapping.py', 'w') as f:
    f.write(mapping_code)

print(f"✓ 映射表已保存到: ./layer_mapping.py")
print(f"✓ 成功映射 {len(mapping)} 个层")

# 保存为 CSV
mapping_df = pd.DataFrame([
    {'Keras Layer': k, 'TFLite Tensor': v} 
    for k, v in mapping.items()
])
mapping_df.to_csv('./results/layer_mapping.csv', index=False)
print(f"✓ CSV 格式已保存到: ./results/layer_mapping.csv")

print("\n"+"="*80)
print("映射统计")
print("="*80)
print(f"Keras 层总数: {len(keras_outputs)}")
print(f"成功映射层数: {len(mapping)}")
print(f"未映射层数: {len(keras_outputs) - len(mapping)}")
