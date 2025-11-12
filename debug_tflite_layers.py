"""
调试TFLite模型层结构
"""
import tensorflow as tf
import numpy as np

tflite_path = './exported_models/tflite/ecgformer_ptq_dynamic.tflite'

print("加载TFLite模型...")
interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

# 获取所有张量详情
tensor_details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n总共有 {len(tensor_details)} 个张量")
print(f"输入张量: {len(input_details)} 个")
print(f"输出张量: {len(output_details)} 个")

print("\n=== 输入详情 ===")
for inp in input_details:
    print(f"  名称: {inp['name']}")
    print(f"  形状: {inp['shape']}")
    print(f"  类型: {inp['dtype']}")

print("\n=== 输出详情 ===")
for out in output_details:
    print(f"  名称: {out['name']}")
    print(f"  形状: {out['shape']}")
    print(f"  类型: {out['dtype']}")

print("\n=== 所有张量（过滤后） ===")
activation_tensors = []
for i, detail in enumerate(tensor_details):
    name = detail['name']
    shape = detail['shape']
    dtype = detail['dtype']

    # 跳过常量
    if 'constant' in name.lower() or 'pseudo' in name.lower():
        continue

    # 检查是否可能是激活层
    # 激活层通常第一维是batch size (1)
    if len(shape) > 0 and shape[0] == 1:
        activation_tensors.append((i, name, shape, dtype))
        print(f"{i:3d}: {name:60s} | Shape: {str(shape):20s} | Type: {dtype}")

print(f"\n找到 {len(activation_tensors)} 个可能的激活张量")

# 测试提取
print("\n=== 测试提取张量值 ===")
# 创建一个假输入
test_input = np.random.randn(1, 187, 1).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()

print("成功运行推理！")

# 尝试提取前10个激活张量
print("\n尝试提取激活张量的值...")
for i, (idx, name, shape, dtype) in enumerate(activation_tensors[:10]):
    try:
        tensor = interpreter.get_tensor(idx)
        print(f"✓ {name:60s} | 提取成功，实际形状: {tensor.shape}")
    except Exception as e:
        print(f"✗ {name:60s} | 提取失败: {str(e)[:50]}")

