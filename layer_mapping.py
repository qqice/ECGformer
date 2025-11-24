# Keras 层名称 → TFLite 张量名称的精确映射表
# 基于 TFLite 命名规则手动构建的准确映射

# LayerNormalization 层的输出是 add_2
# MultiHeadAttention 层的输出包含 attention_output 和后续的 Add
# Conv1D 后面通常是 Add（残差连接）
# Dense 层的输出包含 Relu 和 BiasAdd

LAYER_MAPPING = {
    # 第一个 Transformer Block
    'layer_normalization': 'functional_1/layer_normalization_1/add_2',
    'multi_head_attention': 'functional_1/multi_head_attention_1/attention_output_1/MatMul;functional_1/multi_head_attention_1/attention_output_1/Add',
    'dropout_1': 'functional_1/multi_head_attention_1/attention_output_1/MatMul;functional_1/multi_head_attention_1/attention_output_1/Add',
    'layer_normalization_1': 'functional_1/layer_normalization_1_2/add_2',
    'conv1d': 'functional_1/conv1d_1/Relu;functional_1/conv1d_1/convolution/Squeeze',
    'dropout_2': 'functional_1/conv1d_1/Relu;functional_1/conv1d_1/convolution/Squeeze',
    'conv1d_1': 'functional_1/conv1d_1_2/Add',
    
    # 第二个 Transformer Block
    'layer_normalization_2': 'functional_1/layer_normalization_2_1/add_2',
    'multi_head_attention_1': 'functional_1/multi_head_attention_1_2/attention_output_1/MatMul;functional_1/multi_head_attention_1_2/attention_output_1/Add',
    'dropout_4': 'functional_1/multi_head_attention_1_2/attention_output_1/MatMul;functional_1/multi_head_attention_1_2/attention_output_1/Add',
    'layer_normalization_3': 'functional_1/layer_normalization_3_1/add_2',
    'conv1d_2': 'functional_1/conv1d_2_1/Relu;functional_1/conv1d_2_1/convolution/Squeeze',
    'dropout_5': 'functional_1/conv1d_2_1/Relu;functional_1/conv1d_2_1/convolution/Squeeze',
    'conv1d_3': 'functional_1/conv1d_3_1/Add',
    
    # 第三个 Transformer Block
    'layer_normalization_4': 'functional_1/layer_normalization_4_1/add_2',
    'multi_head_attention_2': 'functional_1/multi_head_attention_2_1/attention_output_1/MatMul;functional_1/multi_head_attention_2_1/attention_output_1/Add',
    'dropout_7': 'functional_1/multi_head_attention_2_1/attention_output_1/MatMul;functional_1/multi_head_attention_2_1/attention_output_1/Add',
    'layer_normalization_5': 'functional_1/layer_normalization_5_1/add_2',
    'conv1d_4': 'functional_1/conv1d_4_1/Relu;functional_1/conv1d_4_1/convolution/Squeeze',
    'dropout_8': 'functional_1/conv1d_4_1/Relu;functional_1/conv1d_4_1/convolution/Squeeze',
    'conv1d_5': 'functional_1/conv1d_5_1/Add',
    
    # 第四个 Transformer Block
    'layer_normalization_6': 'functional_1/layer_normalization_6_1/add_2',
    'multi_head_attention_3': 'functional_1/multi_head_attention_3_1/attention_output_1/MatMul;functional_1/multi_head_attention_3_1/attention_output_1/Add',
    'dropout_10': 'functional_1/multi_head_attention_3_1/attention_output_1/MatMul;functional_1/multi_head_attention_3_1/attention_output_1/Add',
    'layer_normalization_7': 'functional_1/layer_normalization_7_1/add_2',
    'conv1d_6': 'functional_1/conv1d_6_1/Relu;functional_1/conv1d_6_1/convolution/Squeeze',
    'dropout_11': 'functional_1/conv1d_6_1/Relu;functional_1/conv1d_6_1/convolution/Squeeze',
    'conv1d_7': 'functional_1/conv1d_7_1/Add',
    
    # Classification head
    'global_average_pooling1d': 'functional_1/global_average_pooling1d_1/Mean',
    'dense': 'functional_1/dense_1/MatMul;functional_1/dense_1/Relu;functional_1/dense_1/BiasAdd',
    'dropout_12': 'functional_1/dense_1/MatMul;functional_1/dense_1/Relu;functional_1/dense_1/BiasAdd',
    'dense_1': 'functional_1/dense_1_2/MatMul;functional_1/dense_1_2/Relu;functional_1/dense_1_2/BiasAdd',
    'dropout_13': 'functional_1/dense_1_2/MatMul;functional_1/dense_1_2/Relu;functional_1/dense_1_2/BiasAdd',
    'dense_2': 'StatefulPartitionedCall_1:0',  # 最终输出（Softmax后的概率）
}

# 反向映射（TFLite → Keras）
REVERSE_MAPPING = {v: k for k, v in LAYER_MAPPING.items()}

# 量化层标记（仅这些层被量化）
QUANTIZED_LAYERS = {
    'dense',    # 第一个全连接层（128维）
    'dense_1',  # 第二个全连接层（64维）
}

def get_tflite_tensor_name(keras_layer_name):
    """根据 Keras 层名称获取对应的 TFLite 张量名称"""
    return LAYER_MAPPING.get(keras_layer_name, None)

def get_keras_layer_name(tflite_tensor_name):
    """根据 TFLite 张量名称获取对应的 Keras 层名称"""
    return REVERSE_MAPPING.get(tflite_tensor_name, None)

def is_quantized(keras_layer_name):
    """判断某个 Keras 层是否被量化"""
    return keras_layer_name in QUANTIZED_LAYERS
