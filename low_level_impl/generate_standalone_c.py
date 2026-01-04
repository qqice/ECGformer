"""
生成完整独立的ECGformer C实现代码

这个脚本生成一个完整独立的C文件，包含所有权重数据和推理代码。
可以直接编译运行，无需其他头文件。
"""

import numpy as np
import os
import sys
import re
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf


class ECGformerCGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        self.interpreter.allocate_tensors()
        
        self.tensor_details = self.interpreter.get_tensor_details()
        self.tensor_dict = {t['index']: t for t in self.tensor_details}
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 解析操作
        self.ops = self._parse_ops()
        
        # 张量信息和数据
        self.tensors_info = {}
        self.weights_data = {}
        self._load_tensor_info()
        
        # 确定常量和激活张量
        self.constant_tensors = set()
        self.activation_tensors = set()
        self._classify_tensors()
        
    def _parse_ops(self):
        """解析TFLite模型操作"""
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        
        try:
            tf.lite.experimental.Analyzer.analyze(model_path=self.model_path)
        finally:
            sys.stdout = old_stdout
        
        analysis_text = mystdout.getvalue()
        lines = analysis_text.split('\n')
        
        ops = []
        op_pattern = re.compile(r"\s*Op#(\d+)\s+(\w+)\((.*)\)\s*->\s*\[(.*)\]")
        
        for line in lines:
            op_match = op_pattern.match(line)
            if op_match:
                op_id = int(op_match.group(1))
                op_type = op_match.group(2)
                inputs_str = op_match.group(3)
                outputs_str = op_match.group(4)
                
                input_ids = [int(x) for x in re.findall(r"T#(\d+)", inputs_str)]
                output_ids = [int(x) for x in re.findall(r"T#(\d+)", outputs_str)]
                
                ops.append({
                    'id': op_id,
                    'type': op_type,
                    'inputs': input_ids,
                    'outputs': output_ids
                })
        
        return ops
    
    def _load_tensor_info(self):
        """加载所有张量信息"""
        for t in self.tensor_details:
            tid = t['index']
            shape = tuple(t['shape'])
            size = int(np.prod(shape)) if len(shape) > 0 else 1
            
            # 量化参数
            qp = t.get('quantization_parameters', {})
            scales = qp.get('scales', np.array([]))
            zps = qp.get('zero_points', np.array([]))
            qdim = qp.get('quantized_dimension', 0)
            old_q = t.get('quantization', (0.0, 0))
            
            if len(scales) == 0 and old_q[0] != 0:
                scales = np.array([old_q[0]])
                zps = np.array([old_q[1]])
            
            self.tensors_info[tid] = {
                'shape': shape,
                'size': size,
                'scales': scales,
                'zps': zps,
                'qdim': qdim,
                'dtype': str(t['dtype'])
            }
            
            # 尝试获取权重数据
            try:
                data = self.interpreter.get_tensor(tid)
                self.weights_data[tid] = data.copy()
            except:
                pass
    
    def _classify_tensors(self):
        """分类张量：常量vs激活"""
        output_tensors = set()
        for op in self.ops:
            for tid in op['outputs']:
                output_tensors.add(tid)
        
        # 输入张量ID
        input_tid = self.input_details[0]['index']
        
        for op in self.ops:
            for tid in op['inputs']:
                # 排除输入张量和其他操作的输出
                if tid in self.weights_data and tid not in output_tensors and tid != input_tid:
                    self.constant_tensors.add(tid)
            for tid in op['outputs']:
                self.activation_tensors.add(tid)
        
        # 输入张量也是激活张量
        self.activation_tensors.add(input_tid)
        
        # 从常量张量中移除激活张量
        self.constant_tensors -= self.activation_tensors
    
    def _get_scale_zp(self, tid):
        """获取张量的scale和zero_point"""
        info = self.tensors_info.get(tid, {})
        scales = info.get('scales', np.array([]))
        zps = info.get('zps', np.array([]))
        if len(scales) > 0:
            return float(scales[0]), int(zps[0]) if len(zps) > 0 else 0
        return 1.0, 0
    
    def generate(self, output_path: str):
        """生成完整的独立C实现（单文件模式）"""
        
        code = self._generate_header()
        code += self._generate_weights()
        code += self._generate_quant_params()
        code += self._generate_tensor_storage()
        code += self._generate_ops()
        code += self._generate_inference()
        code += self._generate_main()
        
        with open(output_path, 'w') as f:
            f.write(code)
        
        print(f"生成完成: {output_path}")
        print(f"  操作数: {len(self.ops)}")
        print(f"  常量张量: {len(self.constant_tensors)}")
        print(f"  激活张量: {len(self.activation_tensors)}")
    
    def generate_modular(self, output_dir: str):
        """生成模块化的C实现（多文件模式）
        
        生成的文件:
        - ecgformer_config.h    : 模型配置和类型定义
        - ecgformer_weights.h   : INT8权重数据
        - ecgformer_biases.h    : INT32偏置数据
        - ecgformer_quant.h     : 量化参数（scale和zero_point）
        - ecgformer_ops.h       : 操作函数实现
        - ecgformer_model.c     : 主程序（推理函数和main）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 生成配置头文件
        self._generate_config_header(output_dir)
        
        # 2. 生成权重头文件
        self._generate_weights_header(output_dir)
        
        # 3. 生成偏置头文件
        self._generate_biases_header(output_dir)
        
        # 4. 生成量化参数头文件
        self._generate_quant_header(output_dir)
        
        # 5. 生成操作头文件
        self._generate_ops_header(output_dir)
        
        # 6. 生成主程序
        self._generate_main_source(output_dir)
        
        print(f"\n模块化代码生成完成: {output_dir}")
        print(f"  操作数: {len(self.ops)}")
        print(f"  常量张量: {len(self.constant_tensors)}")
        print(f"  激活张量: {len(self.activation_tensors)}")
        print(f"\n生成的文件:")
        print(f"  - ecgformer_config.h  : 模型配置")
        print(f"  - ecgformer_weights.h : 权重数据")
        print(f"  - ecgformer_biases.h  : 偏置数据")
        print(f"  - ecgformer_quant.h   : 量化参数")
        print(f"  - ecgformer_ops.h     : 操作函数")
        print(f"  - ecgformer_model.c   : 主程序")
        print(f"\n编译命令:")
        print(f"  gcc -O3 -o ecgformer ecgformer_model.c -lm")
    
    def _generate_config_header(self, output_dir: str):
        """生成配置头文件"""
        input_shape = tuple(self.input_details[0]['shape'])
        output_shape = tuple(self.output_details[0]['shape'])
        input_scale, input_zp = self._get_scale_zp(self.input_details[0]['index'])
        output_scale, output_zp = self._get_scale_zp(self.output_details[0]['index'])
        
        # 统计张量信息
        total_activation_size = sum(self.tensors_info[tid]['size'] 
                                   for tid in self.activation_tensors 
                                   if tid in self.tensors_info)
        
        code = f'''/**
 * ECGformer INT8 模型配置
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_CONFIG_H
#define ECGFORMER_CONFIG_H

#include <stdint.h>

// ============== 模型配置 ==============

#define INPUT_SIZE {int(np.prod(input_shape))}
#define OUTPUT_CLASSES {output_shape[-1]}
#define NUM_TENSORS {len(self.tensors_info)}
#define ACTIVATION_POOL_SIZE {total_activation_size}

// 输入量化参数
#define INPUT_SCALE {input_scale:.10e}f
#define INPUT_ZERO_POINT {input_zp}

// 输出量化参数
#define OUTPUT_SCALE {output_scale:.10e}f
#define OUTPUT_ZERO_POINT {output_zp}

// 类别名称
static const char* CLASS_NAMES[5] = {{"N (正常)", "S (室上性)", "V (室性)", "F (融合)", "Q (未知)"}};

#endif // ECGFORMER_CONFIG_H
'''
        
        path = os.path.join(output_dir, 'ecgformer_config.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_config.h")
    
    def _generate_weights_header(self, output_dir: str):
        """生成权重头文件（INT8权重）"""
        code = '''/**
 * ECGformer INT8 权重数据
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_WEIGHTS_H
#define ECGFORMER_WEIGHTS_H

#include <stdint.h>

// ============== INT8 权重数据 ==============

'''
        
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            
            data = self.weights_data[tid]
            info = self.tensors_info[tid]
            dtype = info['dtype']
            size = info['size']
            
            # 只处理INT8权重
            if 'int8' in dtype:
                flat = data.flatten().astype(np.int8)
                shape_str = 'x'.join(str(s) for s in info['shape'])
                code += f'// 张量 {tid}: 形状 [{shape_str}]\n'
                code += f'static const int8_t weight_t{tid}[{size}] = {{\n    '
                for i, v in enumerate(flat):
                    code += f'{v}'
                    if i < len(flat) - 1:
                        code += ', '
                        if (i + 1) % 20 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        code += '#endif // ECGFORMER_WEIGHTS_H\n'
        
        path = os.path.join(output_dir, 'ecgformer_weights.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_weights.h")
    
    def _generate_biases_header(self, output_dir: str):
        """生成偏置头文件（INT32偏置）"""
        code = '''/**
 * ECGformer INT32 偏置数据
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_BIASES_H
#define ECGFORMER_BIASES_H

#include <stdint.h>

// ============== INT32 偏置数据 ==============

'''
        
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            
            data = self.weights_data[tid]
            info = self.tensors_info[tid]
            dtype = info['dtype']
            size = info['size']
            
            # 只处理INT32偏置
            if 'int32' in dtype:
                flat = data.flatten().astype(np.int32)
                shape_str = 'x'.join(str(s) for s in info['shape'])
                code += f'// 张量 {tid}: 形状 [{shape_str}]\n'
                code += f'static const int32_t bias_t{tid}[{size}] = {{\n    '
                for i, v in enumerate(flat):
                    code += f'{v}'
                    if i < len(flat) - 1:
                        code += ', '
                        if (i + 1) % 10 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        code += '#endif // ECGFORMER_BIASES_H\n'
        
        path = os.path.join(output_dir, 'ecgformer_biases.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_biases.h")
    
    def _generate_quant_header(self, output_dir: str):
        """生成量化参数头文件"""
        code = '''/**
 * ECGformer 量化参数
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_QUANT_H
#define ECGFORMER_QUANT_H

// ============== Per-Channel 量化参数 ==============

'''
        
        # 为有多通道scale的张量生成scale数组
        for tid in sorted(self.constant_tensors):
            info = self.tensors_info.get(tid, {})
            scales = info.get('scales', np.array([]))
            zps = info.get('zps', np.array([]))
            
            if len(scales) > 1:
                code += f'// 张量 {tid} 的per-channel scales ({len(scales)} channels)\n'
                code += f'static const float scales_t{tid}[{len(scales)}] = {{\n    '
                for i, s in enumerate(scales):
                    code += f'{s:.10e}f'
                    if i < len(scales) - 1:
                        code += ', '
                        if (i + 1) % 4 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        code += '''
// ============== 张量量化信息结构 ==============

typedef struct {
    float scale;
    int32_t zero_point;
} QuantInfo;

'''
        
        # 生成所有张量的量化信息
        code += f'static const QuantInfo tensor_quant[{len(self.tensors_info)}] = {{\n'
        for tid in range(len(self.tensors_info)):
            scale, zp = self._get_scale_zp(tid)
            code += f'    [{tid}] = {{ {scale:.10e}f, {zp} }},\n'
        code += '};\n\n'
        
        code += '#endif // ECGFORMER_QUANT_H\n'
        
        path = os.path.join(output_dir, 'ecgformer_quant.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_quant.h")
    
    def _generate_ops_header(self, output_dir: str):
        """生成操作函数头文件"""
        code = '''/**
 * ECGformer 操作函数
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_OPS_H
#define ECGFORMER_OPS_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

// ============== 辅助函数 ==============

static inline int8_t saturate_int8(int32_t value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static inline int8_t quantize_float(float value, float scale, int32_t zp) {
    return saturate_int8((int32_t)roundf(value / scale) + zp);
}

static inline float dequantize_int8(int8_t value, float scale, int32_t zp) {
    return ((float)value - (float)zp) * scale;
}

// ============== 元素级操作 ==============

// 元素级加法
static void op_add(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float r1 = s1 / so, r2 = s2 / so;
    for (int i = 0; i < size; i++) {
        float v = ((float)in1[i] - z1) * r1 + ((float)in2[i] - z2) * r2;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 元素级减法
static void op_sub(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float r1 = s1 / so, r2 = s2 / so;
    for (int i = 0; i < size; i++) {
        float v = ((float)in1[i] - z1) * r1 - ((float)in2[i] - z2) * r2;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 元素级乘法
static void op_mul(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float eff = (s1 * s2) / so;
    for (int i = 0; i < size; i++) {
        float v = ((float)in1[i] - z1) * ((float)in2[i] - z2) * eff;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 平方差
static void op_squared_diff(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                            float s1, int z1, float s2, int z2, float so, int zo) {
    float eff = (s1 * s1) / so;
    for (int i = 0; i < size; i++) {
        float diff = ((float)in1[i] - z1) - ((float)in2[i] - z2) * (s2 / s1);
        float v = diff * diff * eff;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// ============== 激活函数 ==============

// 倒数平方根
static void op_rsqrt(const int8_t* in, int8_t* out, int size,
                     float si, int zi, float so, int zo) {
    for (int i = 0; i < size; i++) {
        float val = ((float)in[i] - zi) * si;
        float rsqrt = 1.0f / sqrtf(fmaxf(val, 1e-12f));
        out[i] = saturate_int8((int32_t)roundf(rsqrt / so) + zo);
    }
}

// Softmax (沿最后一个维度)
static void op_softmax(const int8_t* input, int8_t* output, int batch, int classes,
                       float si, int zi, float so, int zo) {
    float* vals = (float*)malloc(classes * sizeof(float));
    for (int b = 0; b < batch; b++) {
        float max_val = -1e9f;
        for (int c = 0; c < classes; c++) {
            vals[c] = ((float)input[b*classes + c] - zi) * si;
            if (vals[c] > max_val) max_val = vals[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < classes; c++) {
            vals[c] = expf(vals[c] - max_val);
            sum += vals[c];
        }
        for (int c = 0; c < classes; c++) {
            float softmax_val = vals[c] / sum;
            output[b*classes + c] = saturate_int8((int32_t)roundf(softmax_val / so) + zo);
        }
    }
    free(vals);
}

// ============== 线性操作 ==============

// 全连接层
static void op_fc(const int8_t* input, int batch, int in_dim, int out_dim,
                  const int8_t* weight, const int32_t* bias, int8_t* output,
                  float si, int zi, const float* w_scales, float so, int zo) {
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            int32_t acc = 0;
            for (int i = 0; i < in_dim; i++) {
                acc += ((int32_t)input[b * in_dim + i] - zi) * (int32_t)weight[o * in_dim + i];
            }
            if (bias) acc += bias[o];
            float scale = (si * w_scales[o]) / so;
            output[b * out_dim + o] = saturate_int8((int32_t)roundf(acc * scale) + zo);
        }
    }
}

// 批量矩阵乘法
static void op_batch_matmul(const int8_t* in1, const int8_t* in2, int8_t* out,
                            int batch, int m, int k, int n,
                            float s1, int z1, float s2, int z2, float so, int zo) {
    float eff = (s1 * s2) / so;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int32_t acc = 0;
                for (int l = 0; l < k; l++) {
                    acc += ((int32_t)in1[b*m*k + i*k + l] - z1) * 
                           ((int32_t)in2[b*k*n + l*n + j] - z2);
                }
                out[b*m*n + i*n + j] = saturate_int8((int32_t)roundf(acc * eff) + zo);
            }
        }
    }
}

// ============== 归约操作 ==============

// 均值
static void op_mean(const int8_t* input, int8_t* output,
                    int outer, int reduce_size, int inner,
                    float si, int zi, float so, int zo) {
    float scale_ratio = si / so;
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            int32_t sum = 0;
            for (int r = 0; r < reduce_size; r++) {
                sum += input[o * reduce_size * inner + r * inner + i];
            }
            int32_t mean = sum / reduce_size;
            output[o * inner + i] = saturate_int8((int32_t)roundf((mean - zi) * scale_ratio) + zo);
        }
    }
}

// ============== 形状操作 ==============

// Reshape/复制
static void op_copy(const int8_t* in, int8_t* out, int size) {
    if (in != out) memcpy(out, in, size);
}

// Transpose 3D
static void op_transpose_3d(const int8_t* in, int8_t* out,
                            int d0, int d1, int d2, int p0, int p1, int p2) {
    int dims[3] = {d0, d1, d2};
    int perm[3] = {p0, p1, p2};
    int new_d[3] = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
    
    for (int i0 = 0; i0 < d0; i0++) {
        for (int i1 = 0; i1 < d1; i1++) {
            for (int i2 = 0; i2 < d2; i2++) {
                int in_idx = i0 * d1 * d2 + i1 * d2 + i2;
                int old[3] = {i0, i1, i2};
                int new_idx = old[perm[0]] * new_d[1] * new_d[2] + 
                              old[perm[1]] * new_d[2] + old[perm[2]];
                out[new_idx] = in[in_idx];
            }
        }
    }
}

// Transpose 4D
static void op_transpose_4d(const int8_t* in, int8_t* out,
                            int d0, int d1, int d2, int d3, 
                            int p0, int p1, int p2, int p3) {
    int dims[4] = {d0, d1, d2, d3};
    int perm[4] = {p0, p1, p2, p3};
    int new_d[4] = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
    
    for (int i0 = 0; i0 < d0; i0++) {
        for (int i1 = 0; i1 < d1; i1++) {
            for (int i2 = 0; i2 < d2; i2++) {
                for (int i3 = 0; i3 < d3; i3++) {
                    int in_idx = i0*d1*d2*d3 + i1*d2*d3 + i2*d3 + i3;
                    int old[4] = {i0, i1, i2, i3};
                    int new_idx = old[perm[0]]*new_d[1]*new_d[2]*new_d[3] + 
                                  old[perm[1]]*new_d[2]*new_d[3] + 
                                  old[perm[2]]*new_d[3] + old[perm[3]];
                    out[new_idx] = in[in_idx];
                }
            }
        }
    }
}

#endif // ECGFORMER_OPS_H
'''
        
        path = os.path.join(output_dir, 'ecgformer_ops.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_ops.h")
    
    def _generate_main_source(self, output_dir: str):
        """生成主程序源文件"""
        input_tid = self.input_details[0]['index']
        output_tid = self.output_details[0]['index']
        
        # 计算每个激活张量的偏移
        offsets = {}
        offset = 0
        for tid in sorted(self.activation_tensors):
            if tid in self.tensors_info:
                offsets[tid] = offset
                offset += self.tensors_info[tid]['size']
        
        code = '''/**
 * ECGformer INT8 主程序
 * 自动生成 - 请勿手动修改
 * 
 * 编译: gcc -O3 -o ecgformer ecgformer_model.c -lm
 * 共享库: gcc -O3 -shared -fPIC -DBUILD_SHARED_LIB -o libecgformer.so ecgformer_model.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// 包含头文件
#include "ecgformer_config.h"
#include "ecgformer_weights.h"
#include "ecgformer_biases.h"
#include "ecgformer_quant.h"
#include "ecgformer_ops.h"

// ============== 张量存储 ==============

// 激活张量存储池
static int8_t activation_pool[ACTIVATION_POOL_SIZE];

// 张量指针
static int8_t* tensors[NUM_TENSORS];

// 初始化张量指针
static void init_tensors(void) {
'''
        
        # 常量张量指向静态数组
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            info = self.tensors_info[tid]
            dtype = info['dtype']
            if 'int8' in dtype:
                code += f'    tensors[{tid}] = (int8_t*)weight_t{tid};\n'
            elif 'int32' in dtype:
                code += f'    tensors[{tid}] = (int8_t*)bias_t{tid};\n'
        
        # 激活张量指向存储池
        for tid in sorted(self.activation_tensors):
            if tid in offsets:
                code += f'    tensors[{tid}] = &activation_pool[{offsets[tid]}];\n'
        
        code += '''}\n
// ============== 推理函数 ==============

int ecgformer_inference(const float* input_float, float* output_probs) {
    // 量化输入
    for (int i = 0; i < INPUT_SIZE; i++) {
        tensors[''' + str(input_tid) + '''][i] = quantize_float(input_float[i], INPUT_SCALE, INPUT_ZERO_POINT);
    }
    
'''
        
        # 生成每个操作
        for op in self.ops:
            code += self._generate_op_code_modular(op)
        
        # 输出处理
        code += f'''
    // 反量化输出并找预测类别
    int pred = 0;
    float max_prob = -1e9f;
    for (int i = 0; i < OUTPUT_CLASSES; i++) {{
        output_probs[i] = dequantize_int8(tensors[{output_tid}][i], OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        if (output_probs[i] > max_prob) {{
            max_prob = output_probs[i];
            pred = i;
        }}
    }}
    return pred;
}}

// 获取INT8输出（用于验证）
void ecgformer_get_int8_output(int8_t* output) {{
    memcpy(output, tensors[{output_tid}], OUTPUT_CLASSES);
}}

// ============== 共享库接口 ==============

#ifdef BUILD_SHARED_LIB
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void c_init(void) {{
    init_tensors();
}}

EXPORT int c_inference(const float* input, float* output) {{
    return ecgformer_inference(input, output);
}}

EXPORT void c_get_int8_output(int8_t* output) {{
    ecgformer_get_int8_output(output);
}}
#endif

// ============== 主函数 ==============

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {{
    init_tensors();
    
    printf("ECGformer INT8 模块化C实现\\n");
    printf("==============================\\n");
    
    // 测试用随机输入
    float test_input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {{
        test_input[i] = ((float)rand() / RAND_MAX - 0.5f);
    }}
    
    // 推理
    float output_probs[OUTPUT_CLASSES];
    int pred = ecgformer_inference(test_input, output_probs);
    
    printf("\\n预测结果:\\n");
    for (int i = 0; i < OUTPUT_CLASSES; i++) {{
        printf("  类别 %d (%s): %.4f%s\\n", i, CLASS_NAMES[i], output_probs[i],
               i == pred ? " <-- 预测" : "");
    }}
    
    return 0;
}}
#endif
'''
        
        path = os.path.join(output_dir, 'ecgformer_model.c')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_model.c")
    
    def _generate_op_code_modular(self, op):
        """为单个操作生成代码（模块化版本，使用头文件中的数组名）"""
        op_id = op['id']
        op_type = op['type']
        inputs = op['inputs']
        outputs = op['outputs']
        
        out_tid = outputs[0]
        out_info = self.tensors_info.get(out_tid, {})
        out_size = out_info.get('size', 1)
        out_shape = out_info.get('shape', ())
        out_scale, out_zp = self._get_scale_zp(out_tid)
        
        code = f'    // Op#{op_id}: {op_type}\n'
        
        if op_type == 'RESHAPE':
            in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
            code += f'    op_copy(tensors[{inputs[0]}], tensors[{out_tid}], {in_size});\n'
        
        elif op_type == 'EXPAND_DIMS':
            in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
            code += f'    op_copy(tensors[{inputs[0]}], tensors[{out_tid}], {in_size});\n'
        
        elif op_type == 'TRANSPOSE':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            perm_tid = inputs[1]
            perm = self.weights_data.get(perm_tid, np.arange(len(in_shape))).flatten().tolist()
            
            if len(in_shape) == 3:
                code += f'    op_transpose_3d(tensors[{inputs[0]}], tensors[{out_tid}], '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])});\n'
            elif len(in_shape) == 4:
                code += f'    op_transpose_4d(tensors[{inputs[0]}], tensors[{out_tid}], '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])}, {int(perm[3])});\n'
            else:
                code += f'    op_copy(tensors[{inputs[0]}], tensors[{out_tid}], {out_size});\n'
        
        elif op_type == 'ADD':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            code += f'    op_add(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
            code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SUB':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(tensors[{out_tid}], {out_zp}, {out_size});\n'
            else:
                code += f'    op_sub(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
                code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'MUL':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            code += f'    op_mul(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
            code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SQUARED_DIFFERENCE':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(tensors[{out_tid}], {out_zp}, {out_size});\n'
            else:
                code += f'    op_squared_diff(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
                code += f'                    {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'RSQRT':
            si, zi = self._get_scale_zp(inputs[0])
            code += f'    op_rsqrt(tensors[{inputs[0]}], tensors[{out_tid}], {out_size},\n'
            code += f'             {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'FULLY_CONNECTED':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            in_dim = in_shape[-1] if len(in_shape) > 0 else 1
            batch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            out_dim = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            bias_str = f'(const int32_t*)tensors[{inputs[2]}]' if has_bias else 'NULL'
            
            # 检查是否有per-channel scales
            weight_info = self.tensors_info.get(weight_tid, {})
            weight_scales = weight_info.get('scales', np.array([]))
            
            if len(weight_scales) > 1:
                scales_str = f'scales_t{weight_tid}'
            else:
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                code += f'    {{ float ws[{out_dim}]; for(int i=0;i<{out_dim};i++) ws[i]={single_scale:.10e}f;\n'
                scales_str = 'ws'
            
            code += f'    op_fc(tensors[{inputs[0]}], {batch}, {in_dim}, {out_dim},\n'
            code += f'          (const int8_t*)tensors[{weight_tid}], {bias_str}, tensors[{out_tid}],\n'
            code += f'          {si:.10e}f, {zi}, {scales_str}, {out_scale:.10e}f, {out_zp});\n'
            
            if len(weight_scales) <= 1:
                code += '    }\n'
        
        elif op_type == 'CONV_2D':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            c_in = in_shape[-1] if len(in_shape) > 0 else 1
            spatial = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            c_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            bias_str = f'(const int32_t*)tensors[{inputs[2]}]' if has_bias else 'NULL'
            
            weight_info = self.tensors_info.get(weight_tid, {})
            weight_scales = weight_info.get('scales', np.array([]))
            
            if len(weight_scales) > 1:
                scales_str = f'scales_t{weight_tid}'
            else:
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                code += f'    {{ float ws[{c_out}]; for(int i=0;i<{c_out};i++) ws[i]={single_scale:.10e}f;\n'
                scales_str = 'ws'
            
            code += f'    op_fc(tensors[{inputs[0]}], {spatial}, {c_in}, {c_out},\n'
            code += f'          (const int8_t*)tensors[{weight_tid}], {bias_str}, tensors[{out_tid}],\n'
            code += f'          {si:.10e}f, {zi}, {scales_str}, {out_scale:.10e}f, {out_zp});\n'
            
            if len(weight_scales) <= 1:
                code += '    }\n'
        
        elif op_type == 'BATCH_MATMUL':
            in1_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            in2_shape = self.tensors_info.get(inputs[1], {}).get('shape', ())
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            
            batch = in1_shape[0] if len(in1_shape) > 2 else 1
            m = in1_shape[1] if len(in1_shape) > 1 else 1
            k = in1_shape[2] if len(in1_shape) > 2 else in1_shape[1] if len(in1_shape) > 1 else 1
            n = in2_shape[2] if len(in2_shape) > 2 else in2_shape[1] if len(in2_shape) > 1 else 1
            
            code += f'    op_batch_matmul(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}],\n'
            code += f'                    {batch}, {m}, {k}, {n},\n'
            code += f'                    {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'MEAN':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            
            outer = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            reduce_size = in_shape[-1] if len(in_shape) > 0 else 1
            inner = 1
            
            code += f'    op_mean(tensors[{inputs[0]}], tensors[{out_tid}],\n'
            code += f'            {outer}, {reduce_size}, {inner},\n'
            code += f'            {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SOFTMAX':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            batch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            classes = in_shape[-1] if len(in_shape) > 0 else 1
            
            code += f'    op_softmax(tensors[{inputs[0]}], tensors[{out_tid}], {batch}, {classes},\n'
            code += f'               {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        else:
            code += f'    // TODO: 实现 {op_type}\n'
        
        code += '\n'
        return code
    
    def _generate_header(self):
        """生成文件头"""
        input_shape = tuple(self.input_details[0]['shape'])
        output_shape = tuple(self.output_details[0]['shape'])
        input_scale, input_zp = self._get_scale_zp(self.input_details[0]['index'])
        output_scale, output_zp = self._get_scale_zp(self.output_details[0]['index'])
        
        return f'''/**
 * ECGformer INT8 完整C实现
 * 自动生成 - 纯C实现，无外部依赖
 * 
 * 编译: gcc -O3 -o ecgformer ecgformer_standalone.c -lm
 * 运行: ./ecgformer
 * 或者编译为共享库供Python调用: gcc -O3 -shared -fPIC -DBUILD_SHARED_LIB -o libecgformer.so ecgformer_standalone.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// ============== 模型配置 ==============

#define INPUT_SIZE {int(np.prod(input_shape))}
#define OUTPUT_CLASSES {output_shape[-1]}
#define NUM_TENSORS {len(self.tensors_info)}

#define INPUT_SCALE {input_scale:.10e}f
#define INPUT_ZERO_POINT {input_zp}
#define OUTPUT_SCALE {output_scale:.10e}f
#define OUTPUT_ZERO_POINT {output_zp}

static const char* CLASS_NAMES[5] = {{"N (正常)", "S (室上性)", "V (室性)", "F (融合)", "Q (未知)"}};

'''
    
    def _generate_weights(self):
        """生成权重数据"""
        code = '// ============== 权重数据 ==============\n\n'
        
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            
            data = self.weights_data[tid]
            info = self.tensors_info[tid]
            dtype = info['dtype']
            size = info['size']
            
            if 'int32' in dtype:
                # INT32 偏置
                flat = data.flatten().astype(np.int32)
                code += f'static const int32_t const_t{tid}[{size}] = {{\n    '
                for i, v in enumerate(flat):
                    code += f'{v}'
                    if i < len(flat) - 1:
                        code += ', '
                        if (i + 1) % 10 == 0:
                            code += '\n    '
                code += '\n};\n\n'
            else:
                # INT8 权重
                flat = data.flatten().astype(np.int8)
                code += f'static const int8_t const_t{tid}[{size}] = {{\n    '
                for i, v in enumerate(flat):
                    code += f'{v}'
                    if i < len(flat) - 1:
                        code += ', '
                        if (i + 1) % 20 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        return code
    
    def _generate_quant_params(self):
        """生成量化参数"""
        code = '// ============== 量化参数 ==============\n\n'
        
        # 为有多通道scale的张量生成scale数组
        for tid in sorted(self.constant_tensors):
            info = self.tensors_info.get(tid, {})
            scales = info.get('scales', np.array([]))
            if len(scales) > 1:
                code += f'static const float scales_t{tid}[{len(scales)}] = {{\n    '
                for i, s in enumerate(scales):
                    code += f'{s:.10e}f'
                    if i < len(scales) - 1:
                        code += ', '
                        if (i + 1) % 4 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        return code
    
    def _generate_tensor_storage(self):
        """生成张量存储"""
        # 计算激活张量总大小
        total_size = sum(self.tensors_info[tid]['size'] 
                        for tid in self.activation_tensors 
                        if tid in self.tensors_info)
        
        # 计算每个激活张量的偏移
        offsets = {}
        offset = 0
        for tid in sorted(self.activation_tensors):
            if tid in self.tensors_info:
                offsets[tid] = offset
                offset += self.tensors_info[tid]['size']
        
        code = f'''// ============== 张量存储 ==============

// 激活张量存储池 ({total_size} bytes)
static int8_t activation_pool[{total_size}];

// 张量指针
static int8_t* tensors[NUM_TENSORS];

// 初始化张量指针
static void init_tensors(void) {{
'''
        
        # 常量张量指向静态数组
        for tid in sorted(self.constant_tensors):
            if tid in self.weights_data:
                code += f'    tensors[{tid}] = (int8_t*)const_t{tid};\n'
        
        # 激活张量指向存储池
        for tid in sorted(self.activation_tensors):
            if tid in offsets:
                code += f'    tensors[{tid}] = &activation_pool[{offsets[tid]}];\n'
        
        code += '}\n\n'
        return code
    
    def _generate_ops(self):
        """生成操作函数"""
        return '''// ============== 操作实现 ==============

static inline int8_t saturate_int8(int32_t value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static inline int8_t quantize_float(float value, float scale, int32_t zp) {
    return saturate_int8((int32_t)roundf(value / scale) + zp);
}

static inline float dequantize_int8(int8_t value, float scale, int32_t zp) {
    return ((float)value - (float)zp) * scale;
}

// 元素级加法
static void op_add(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float r1 = s1 / so, r2 = s2 / so;
    for (int i = 0; i < size; i++) {
        float v = ((float)in1[i] - z1) * r1 + ((float)in2[i] - z2) * r2;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 元素级减法
static void op_sub(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float r1 = s1 / so, r2 = s2 / so;
    for (int i = 0; i < size; i++) {
        float v = ((float)in1[i] - z1) * r1 - ((float)in2[i] - z2) * r2;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 元素级乘法
static void op_mul(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float eff = (s1 * s2) / so;
    for (int i = 0; i < size; i++) {
        float v = ((float)in1[i] - z1) * ((float)in2[i] - z2) * eff;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 平方差
static void op_squared_diff(const int8_t* in1, const int8_t* in2, int8_t* out, int size,
                            float s1, int z1, float s2, int z2, float so, int zo) {
    float eff = (s1 * s1) / so;
    for (int i = 0; i < size; i++) {
        float diff = ((float)in1[i] - z1) - ((float)in2[i] - z2) * (s2 / s1);
        float v = diff * diff * eff;
        out[i] = saturate_int8((int32_t)roundf(v) + zo);
    }
}

// 倒数平方根
static void op_rsqrt(const int8_t* in, int8_t* out, int size,
                     float si, int zi, float so, int zo) {
    for (int i = 0; i < size; i++) {
        float val = ((float)in[i] - zi) * si;
        float rsqrt = 1.0f / sqrtf(fmaxf(val, 1e-12f));
        out[i] = saturate_int8((int32_t)roundf(rsqrt / so) + zo);
    }
}

// 全连接层
static void op_fc(const int8_t* input, int batch, int in_dim, int out_dim,
                  const int8_t* weight, const int32_t* bias, int8_t* output,
                  float si, int zi, const float* w_scales, float so, int zo) {
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            int32_t acc = 0;
            for (int i = 0; i < in_dim; i++) {
                acc += ((int32_t)input[b * in_dim + i] - zi) * (int32_t)weight[o * in_dim + i];
            }
            if (bias) acc += bias[o];
            float scale = (si * w_scales[o]) / so;
            output[b * out_dim + o] = saturate_int8((int32_t)roundf(acc * scale) + zo);
        }
    }
}

// 批量矩阵乘法
static void op_batch_matmul(const int8_t* in1, const int8_t* in2, int8_t* out,
                            int batch, int m, int k, int n,
                            float s1, int z1, float s2, int z2, float so, int zo) {
    float eff = (s1 * s2) / so;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int32_t acc = 0;
                for (int l = 0; l < k; l++) {
                    acc += ((int32_t)in1[b*m*k + i*k + l] - z1) * 
                           ((int32_t)in2[b*k*n + l*n + j] - z2);
                }
                out[b*m*n + i*n + j] = saturate_int8((int32_t)roundf(acc * eff) + zo);
            }
        }
    }
}

// 均值
static void op_mean(const int8_t* input, int8_t* output,
                    int outer, int reduce_size, int inner,
                    float si, int zi, float so, int zo) {
    float scale_ratio = si / so;
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            int32_t sum = 0;
            for (int r = 0; r < reduce_size; r++) {
                sum += input[o * reduce_size * inner + r * inner + i];
            }
            int32_t mean = sum / reduce_size;
            output[o * inner + i] = saturate_int8((int32_t)roundf((mean - zi) * scale_ratio) + zo);
        }
    }
}

// Softmax (沿最后一个维度)
static void op_softmax(const int8_t* input, int8_t* output, int batch, int classes,
                       float si, int zi, float so, int zo) {
    float* vals = (float*)malloc(classes * sizeof(float));
    for (int b = 0; b < batch; b++) {
        float max_val = -1e9f;
        for (int c = 0; c < classes; c++) {
            vals[c] = ((float)input[b*classes + c] - zi) * si;
            if (vals[c] > max_val) max_val = vals[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < classes; c++) {
            vals[c] = expf(vals[c] - max_val);
            sum += vals[c];
        }
        for (int c = 0; c < classes; c++) {
            float softmax_val = vals[c] / sum;
            output[b*classes + c] = saturate_int8((int32_t)roundf(softmax_val / so) + zo);
        }
    }
    free(vals);
}

// Reshape/复制
static void op_copy(const int8_t* in, int8_t* out, int size) {
    if (in != out) memcpy(out, in, size);
}

// Transpose 3D
static void op_transpose_3d(const int8_t* in, int8_t* out,
                            int d0, int d1, int d2, int p0, int p1, int p2) {
    int dims[3] = {d0, d1, d2};
    int perm[3] = {p0, p1, p2};
    int new_d[3] = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
    
    for (int i0 = 0; i0 < d0; i0++) {
        for (int i1 = 0; i1 < d1; i1++) {
            for (int i2 = 0; i2 < d2; i2++) {
                int in_idx = i0 * d1 * d2 + i1 * d2 + i2;
                int old[3] = {i0, i1, i2};
                int new_idx = old[perm[0]] * new_d[1] * new_d[2] + 
                              old[perm[1]] * new_d[2] + old[perm[2]];
                out[new_idx] = in[in_idx];
            }
        }
    }
}

// Transpose 4D
static void op_transpose_4d(const int8_t* in, int8_t* out,
                            int d0, int d1, int d2, int d3, 
                            int p0, int p1, int p2, int p3) {
    int dims[4] = {d0, d1, d2, d3};
    int perm[4] = {p0, p1, p2, p3};
    int new_d[4] = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
    
    for (int i0 = 0; i0 < d0; i0++) {
        for (int i1 = 0; i1 < d1; i1++) {
            for (int i2 = 0; i2 < d2; i2++) {
                for (int i3 = 0; i3 < d3; i3++) {
                    int in_idx = i0*d1*d2*d3 + i1*d2*d3 + i2*d3 + i3;
                    int old[4] = {i0, i1, i2, i3};
                    int new_idx = old[perm[0]]*new_d[1]*new_d[2]*new_d[3] + 
                                  old[perm[1]]*new_d[2]*new_d[3] + 
                                  old[perm[2]]*new_d[3] + old[perm[3]];
                    out[new_idx] = in[in_idx];
                }
            }
        }
    }
}

'''
    
    def _generate_inference(self):
        """生成推理函数"""
        input_tid = self.input_details[0]['index']
        output_tid = self.output_details[0]['index']
        input_scale, input_zp = self._get_scale_zp(input_tid)
        output_scale, output_zp = self._get_scale_zp(output_tid)
        
        code = '''// ============== 推理函数 ==============

int ecgformer_inference(const float* input_float, float* output_probs) {
    // 量化输入
    for (int i = 0; i < INPUT_SIZE; i++) {
        tensors[''' + str(input_tid) + '''][i] = quantize_float(input_float[i], INPUT_SCALE, INPUT_ZERO_POINT);
    }
    
'''
        
        # 生成每个操作
        for op in self.ops:
            code += self._generate_op_code(op)
        
        # 输出处理
        code += f'''
    // 反量化输出并找预测类别
    int pred = 0;
    float max_prob = -1e9f;
    for (int i = 0; i < OUTPUT_CLASSES; i++) {{
        output_probs[i] = dequantize_int8(tensors[{output_tid}][i], OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        if (output_probs[i] > max_prob) {{
            max_prob = output_probs[i];
            pred = i;
        }}
    }}
    return pred;
}}

// 获取INT8输出（用于验证）
void ecgformer_get_int8_output(int8_t* output) {{
    memcpy(output, tensors[{output_tid}], OUTPUT_CLASSES);
}}

'''
        return code
    
    def _generate_op_code(self, op):
        """为单个操作生成代码"""
        op_id = op['id']
        op_type = op['type']
        inputs = op['inputs']
        outputs = op['outputs']
        
        out_tid = outputs[0]
        out_info = self.tensors_info.get(out_tid, {})
        out_size = out_info.get('size', 1)
        out_shape = out_info.get('shape', ())
        out_scale, out_zp = self._get_scale_zp(out_tid)
        
        code = f'    // Op#{op_id}: {op_type}\n'
        
        if op_type == 'RESHAPE':
            in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
            code += f'    op_copy(tensors[{inputs[0]}], tensors[{out_tid}], {in_size});\n'
        
        elif op_type == 'EXPAND_DIMS':
            in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
            code += f'    op_copy(tensors[{inputs[0]}], tensors[{out_tid}], {in_size});\n'
        
        elif op_type == 'TRANSPOSE':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            perm_tid = inputs[1]
            perm = self.weights_data.get(perm_tid, np.arange(len(in_shape))).flatten().tolist()
            
            if len(in_shape) == 3:
                code += f'    op_transpose_3d(tensors[{inputs[0]}], tensors[{out_tid}], '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])});\n'
            elif len(in_shape) == 4:
                code += f'    op_transpose_4d(tensors[{inputs[0]}], tensors[{out_tid}], '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])}, {int(perm[3])});\n'
            else:
                code += f'    op_copy(tensors[{inputs[0]}], tensors[{out_tid}], {out_size});\n'
        
        elif op_type == 'ADD':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            code += f'    op_add(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
            code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SUB':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(tensors[{out_tid}], {out_zp}, {out_size});\n'
            else:
                code += f'    op_sub(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
                code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'MUL':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            code += f'    op_mul(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
            code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SQUARED_DIFFERENCE':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(tensors[{out_tid}], {out_zp}, {out_size});\n'
            else:
                code += f'    op_squared_diff(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}], {out_size},\n'
                code += f'                    {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'RSQRT':
            si, zi = self._get_scale_zp(inputs[0])
            code += f'    op_rsqrt(tensors[{inputs[0]}], tensors[{out_tid}], {out_size},\n'
            code += f'             {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'FULLY_CONNECTED':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            in_dim = in_shape[-1] if len(in_shape) > 0 else 1
            batch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            out_dim = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            bias_str = f'(const int32_t*)tensors[{inputs[2]}]' if has_bias else 'NULL'
            
            # 检查是否有per-channel scales
            weight_info = self.tensors_info.get(weight_tid, {})
            weight_scales = weight_info.get('scales', np.array([]))
            
            if len(weight_scales) > 1:
                scales_str = f'scales_t{weight_tid}'
            else:
                # 使用单一scale，创建临时数组
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                code += f'    {{ float ws[{out_dim}]; for(int i=0;i<{out_dim};i++) ws[i]={single_scale:.10e}f;\n'
                scales_str = 'ws'
            
            code += f'    op_fc(tensors[{inputs[0]}], {batch}, {in_dim}, {out_dim},\n'
            code += f'          (const int8_t*)tensors[{weight_tid}], {bias_str}, tensors[{out_tid}],\n'
            code += f'          {si:.10e}f, {zi}, {scales_str}, {out_scale:.10e}f, {out_zp});\n'
            
            if len(weight_scales) <= 1:
                code += '    }\n'
        
        elif op_type == 'CONV_2D':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            c_in = in_shape[-1] if len(in_shape) > 0 else 1
            spatial = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            c_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            bias_str = f'(const int32_t*)tensors[{inputs[2]}]' if has_bias else 'NULL'
            
            weight_info = self.tensors_info.get(weight_tid, {})
            weight_scales = weight_info.get('scales', np.array([]))
            
            if len(weight_scales) > 1:
                scales_str = f'scales_t{weight_tid}'
            else:
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                code += f'    {{ float ws[{c_out}]; for(int i=0;i<{c_out};i++) ws[i]={single_scale:.10e}f;\n'
                scales_str = 'ws'
            
            code += f'    op_fc(tensors[{inputs[0]}], {spatial}, {c_in}, {c_out},\n'
            code += f'          (const int8_t*)tensors[{weight_tid}], {bias_str}, tensors[{out_tid}],\n'
            code += f'          {si:.10e}f, {zi}, {scales_str}, {out_scale:.10e}f, {out_zp});\n'
            
            if len(weight_scales) <= 1:
                code += '    }\n'
        
        elif op_type == 'BATCH_MATMUL':
            in1_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            in2_shape = self.tensors_info.get(inputs[1], {}).get('shape', ())
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            
            batch = in1_shape[0] if len(in1_shape) > 2 else 1
            m = in1_shape[1] if len(in1_shape) > 1 else 1
            k = in1_shape[2] if len(in1_shape) > 2 else in1_shape[1] if len(in1_shape) > 1 else 1
            n = in2_shape[2] if len(in2_shape) > 2 else in2_shape[1] if len(in2_shape) > 1 else 1
            
            code += f'    op_batch_matmul(tensors[{inputs[0]}], tensors[{inputs[1]}], tensors[{out_tid}],\n'
            code += f'                    {batch}, {m}, {k}, {n},\n'
            code += f'                    {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'MEAN':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            
            # 假设沿最后一个轴
            outer = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            reduce_size = in_shape[-1] if len(in_shape) > 0 else 1
            inner = 1
            
            code += f'    op_mean(tensors[{inputs[0]}], tensors[{out_tid}],\n'
            code += f'            {outer}, {reduce_size}, {inner},\n'
            code += f'            {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SOFTMAX':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            # softmax沿最后一个维度，所以batch是前面所有维度的乘积
            batch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            classes = in_shape[-1] if len(in_shape) > 0 else 1
            
            code += f'    op_softmax(tensors[{inputs[0]}], tensors[{out_tid}], {batch}, {classes},\n'
            code += f'               {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        else:
            code += f'    // TODO: 实现 {op_type}\n'
        
        code += '\n'
        return code
    
    def _generate_main(self):
        """生成main函数和共享库接口"""
        return '''// ============== 共享库接口 ==============

#ifdef BUILD_SHARED_LIB
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void c_init(void) {
    init_tensors();
}

EXPORT int c_inference(const float* input, float* output) {
    return ecgformer_inference(input, output);
}

EXPORT void c_get_int8_output(int8_t* output) {
    ecgformer_get_int8_output(output);
}
#endif

// ============== 主函数 ==============

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {
    init_tensors();
    
    printf("ECGformer INT8 完整C实现\\n");
    printf("==============================\\n");
    
    // 测试用随机输入
    float test_input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        test_input[i] = ((float)rand() / RAND_MAX - 0.5f);
    }
    
    // 推理
    float output_probs[OUTPUT_CLASSES];
    int pred = ecgformer_inference(test_input, output_probs);
    
    printf("\\n预测结果:\\n");
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        printf("  类别 %d (%s): %.4f%s\\n", i, CLASS_NAMES[i], output_probs[i],
               i == pred ? " <-- 预测" : "");
    }
    
    return 0;
}
#endif
'''


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成ECGformer C实现代码')
    parser.add_argument('--modular', '-m', action='store_true',
                        help='生成模块化多文件版本（默认为单文件版本）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出路径（单文件模式为.c文件路径，模块化模式为目录路径）')
    args = parser.parse_args()
    
    model_path = os.path.join(PROJECT_ROOT, 'exported_models', 'tflite', 
                              'ecgformer_custom_ln_int8.tflite')
    
    print("="*60)
    print("生成ECGformer C实现")
    print("="*60)
    
    generator = ECGformerCGenerator(model_path)
    
    if args.modular:
        output_dir = args.output or os.path.join(SCRIPT_DIR, 'c_export_modular')
        print(f"\n模式: 模块化多文件")
        print(f"输出目录: {output_dir}\n")
        generator.generate_modular(output_dir)
    else:
        output_path = args.output or os.path.join(SCRIPT_DIR, 'c_export', 'ecgformer_standalone.c')
        print(f"\n模式: 单文件独立版")
        print(f"输出文件: {output_path}\n")
        generator.generate(output_path)
