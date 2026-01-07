# -*- coding: utf-8 -*-
"""
生成完整独立的ECGformer C实现代码 - Bare-metal Hardware Verification Style

这个脚本生成一个完整独立的C文件，包含所有权重数据和推理代码。
可以直接编译运行，无需其他头文件。
针对RISC-V AI加速器硬件验证优化。
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
        
        # 预扫描收集 per-channel Q16 scales (适配INT26)
        self._precompute_q16_scales()
    
    def _precompute_q16_scales(self):
        """预扫描所有操作，收集 per-channel Q16 scales (适配26位中间结果)"""
        self.q16_scales = {}  # {op_index: {'weight_tid': tid, 'scales_q16': [...]}}
        
        for op in self.ops:
            op_type = op['type']
            op_idx = op['id']
            inputs = op['inputs']
            outputs = op['outputs']
            
            if op_type in ('FULLY_CONNECTED', 'CONV_2D'):
                if len(inputs) < 2:
                    continue
                    
                weight_tid = inputs[1]
                out_tid = outputs[0] if len(outputs) > 0 else -1
                
                # 获取输入和输出的 scale
                si, zi = self._get_scale_zp(inputs[0])
                so, zo = self._get_scale_zp(out_tid)
                
                # 获取权重的 per-channel scales
                weight_info = self.tensors_info.get(weight_tid, {})
                weight_scales = weight_info.get('scales', np.array([]))
                
                if len(weight_scales) > 1:
                    # 预计算 Q16: scale_q16[o] = (si * weight_scale[o] / so) * (1<<16)
                    # Q16 适配 INT26 中间结果: acc(~20bit) * scale(~16bit) >> 16 = ~20bit
                    scales_q16 = []
                    for ws in weight_scales:
                        eff = (si * ws / so) * (1 << 16)
                        scales_q16.append(int(round(eff)))
                    
                    self.q16_scales[op_idx] = {
                        'weight_tid': weight_tid,
                        'scales_q16': scales_q16,
                        'chn_out': len(weight_scales)
                    }

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
        """生成量化参数头文件 - Bare-metal Style (无结构体, Q16格式适配INT26)"""
        code = '''/**
 * ECGformer 量化参数 - Bare-metal Style
 * 自动生成 - 请勿手动修改
 * 
 * 注意: 不使用结构体, 使用扁平数组以便直接硬件访问
 * Q16格式: scale = (int32_t)((si * weight_scale / so) * (1<<16))
 * 适配INT26中间结果: acc(~20bit) * scale(~16bit) = ~36bit, >> 16后 = ~20bit
 */

#ifndef ECGFORMER_QUANT_H
#define ECGFORMER_QUANT_H

#include <stdint.h>

// ============== 预计算的 Per-Channel Q16 Scales ==============
// Q16格式: 已预计算 (si * weight_scale[o] / so) * (1<<16)
// 每个操作有独立的 Q16 scale 数组 (以操作索引命名)
// 使用 int32_t 适配嵌入式硬件

'''
        
        # 生成预计算的 Q16 scales (以操作索引命名)
        for op_idx in sorted(self.q16_scales.keys()):
            info = self.q16_scales[op_idx]
            scales_q16 = info['scales_q16']
            chn_out = info['chn_out']
            weight_tid = info['weight_tid']
            
            code += f'// Op#{op_idx} 的 per-channel Q16 scales (权重张量 {weight_tid}, {chn_out} channels)\n'
            code += f'static const int32_t pscales_q16_op{op_idx}[{chn_out}] = {{\n    '
            for i, sq16 in enumerate(scales_q16):
                code += f'{sq16}'
                if i < chn_out - 1:
                    code += ', '
                    if (i + 1) % 4 == 0:
                        code += '\n    '
            code += '\n};\n\n'
        
        code += '''
#endif // ECGFORMER_QUANT_H
'''
        
        path = os.path.join(output_dir, 'ecgformer_quant.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_quant.h")
    
    def _generate_ops_header(self, output_dir: str):
        """生成操作函数头文件 - Bare-metal Hardware Verification Style"""
        code = '''/**
 * ECGformer 操作函数 - Bare-metal Hardware Verification Style
 * 自动生成 - 请勿手动修改
 * 
 * 命名约定:
 *   - 指针: p前缀 (pinput, poutput, pfilter)
 *   - 维度: side (inside, oside, fside)
 *   - 通道: chn_in, chn_out
 *   - 临时变量: t0, t1, t2... (模拟寄存器)
 */

#ifndef ECGFORMER_OPS_H
#define ECGFORMER_OPS_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

// ============== 硬件验证宏 ==============

// 位移替代除法 (仅用于2的幂次)
#define DIV2(x)   ((x) >> 1)
#define DIV4(x)   ((x) >> 2)
#define DIV8(x)   ((x) >> 3)
#define DIV16(x)  ((x) >> 4)
#define DIV32(x)  ((x) >> 5)
#define DIV64(x)  ((x) >> 6)
#define DIV128(x) ((x) >> 7)
#define DIV256(x) ((x) >> 8)

// 位移替代乘法 (仅用于2的幂次)
#define MUL2(x)   ((x) << 1)
#define MUL4(x)   ((x) << 2)
#define MUL8(x)   ((x) << 3)
#define MUL16(x)  ((x) << 4)
#define MUL32(x)  ((x) << 5)
#define MUL64(x)  ((x) << 6)
#define MUL128(x) ((x) << 7)
#define MUL256(x) ((x) << 8)

// 自定义指令占位宏 (RISC-V)
#define HW_BARRIER()  asm volatile("" ::: "memory")
#define HW_NOP()      asm volatile("nop")

// 自定义加速器指令模板 (.insn format)
// 用法: HW_CUSTOM_OP(rd, rs1, rs2) 执行自定义操作
// .insn r opcode, funct3, funct7, rd, rs1, rs2
#define HW_MAC_INIT(acc)      asm volatile(".insn r 0x0b, 0x0, 0x00, %0, x0, x0" : "=r"(acc))
#define HW_MAC_ACC(acc, a, b) asm volatile(".insn r 0x0b, 0x1, 0x00, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))
#define HW_QUANT(out, in, s)  asm volatile(".insn r 0x0b, 0x2, 0x00, %0, %1, %2" : "=r"(out) : "r"(in), "r"(s))

// ============== 辅助函数 ==============
// Q16 定点格式: scale = (int32_t)((1.0 / float_scale) * (1 << 16))
// 量化公式: out = ((acc * scale) >> 16) + zp
// 适配INT26中间结果: 使用纯int32_t运算

static inline int8_t saturate_int8(int32_t t0) {
    if (t0 > 127) return 127;
    if (t0 < -128) return -128;
    return (int8_t)t0;
}

// Q16 定点量化: out = ((val * scale) >> 16) + zp
// 注意: 使用分步计算避免溢出, 适配INT26硬件
static inline int8_t quantize_q16(int32_t t0, int32_t scale, int32_t zp) {
    // 分步右移: 先右移8位, 乘scale, 再右移8位
    int32_t t1 = (t0 >> 8) * scale;  // ~18bit * ~16bit = ~34bit, 但先右移8位控制在26位内
    int32_t t2 = t1 >> 8;            // 总共右移16位
    return saturate_int8(t2 + zp);
}

// 浮点输入量化 (仅用于外部接口，计算路径使用Q16)
static inline int8_t quantize_float(float t0, float scale, int32_t zp) {
    int32_t t1 = (int32_t)roundf(t0 / scale) + zp;
    return saturate_int8(t1);
}

// 反量化 (仅用于外部接口/调试)
static inline float dequantize_int8(int8_t t0, float scale, int32_t zp) {
    return ((float)t0 - (float)zp) * scale;
}

// ============== 元素级操作 ==============
// Q16格式: scale参数 = (int32_t)((1.0/float_scale) * (1<<16))

// 元素级加法: pout[i] = pin1[i] + pin2[i] (Q16定点)
// scale1 = (s1/so) * (1<<16), scale2 = (s2/so) * (1<<16)
static void op_add(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   int32_t scale1, int z1, int32_t scale2, int z2, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        // Q16加法: ((t0 * scale1) + (t1 * scale2)) >> 16
        int32_t t2 = (t0 * scale1 + t1 * scale2) >> 16;
        *(pout + i) = saturate_int8(t2 + zo);
    }
}

// 元素级减法: pout[i] = pin1[i] - pin2[i] (Q16定点)
// scale1 = (s1/so) * (1<<16), scale2 = (s2/so) * (1<<16)
static void op_sub(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   int32_t scale1, int z1, int32_t scale2, int z2, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        // Q16减法: ((t0 * scale1) - (t1 * scale2)) >> 16
        int32_t t2 = (t0 * scale1 - t1 * scale2) >> 16;
        *(pout + i) = saturate_int8(t2 + zo);
    }
}

// 元素级乘法: pout[i] = pin1[i] * pin2[i] (Q16定点)
// scale = (s1 * s2 / so) * (1<<16)
static void op_mul(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   int32_t scale, int z1, int z2, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        int32_t t2 = t0 * t1;  // 累积乘积 (~16bit)
        // Q16量化: (t2 * scale) >> 16, 分步计算避免溢出
        int32_t t3 = ((t2 >> 8) * scale) >> 8;
        *(pout + i) = saturate_int8(t3 + zo);
    }
}

// 平方差: pout[i] = (pin1[i] - pin2[i])^2 (Q16定点)
// scale_diff = (s2/s1) * (1<<16), scale_out = (s1*s1/so) * (1<<16)
static void op_squared_diff(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                            int32_t scale_diff, int z1, int z2, int32_t scale_out, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        // 先计算差值 (Q16): t0 - (t1 * scale_diff) >> 16
        int32_t t2 = (t1 * scale_diff) >> 16;
        int32_t t3 = t0 - t2;
        // 平方后量化: (t3 * t3 * scale_out) >> 16, 分步计算
        int32_t t4 = ((t3 * t3) >> 8);
        int32_t t5 = (t4 * scale_out) >> 8;
        *(pout + i) = saturate_int8(t5 + zo);
    }
}

// ============== 激活函数 ==============

// 倒数平方根: pout[i] = 1/sqrt(pin[i])
// 注: rsqrt核心需要浮点, 量化输入/输出使用Q22
// scale_in = si * (1<<22), scale_out = (1/so) * (1<<22)
static void op_rsqrt(const int8_t* pin, int8_t* pout, int len,
                     float scale_in, int zi, float scale_out, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin + i) - zi;
        float t1 = (float)t0 * scale_in;  // 反量化
        if (t1 <= 0) t1 = 1e-10f;  // 防止除零
        float t2 = 1.0f / sqrtf(t1);  // rsqrt
        int32_t t3 = (int32_t)roundf(t2 / scale_out) + zo;  // 量化输出
        *(pout + i) = saturate_int8(t3);
    }
}

// Softmax: 沿最后一个维度
// 注: softmax核心需要exp, 使用浮点计算, 量化输入/输出
static void op_softmax(const int8_t* pin, int8_t* pout, int nbatch, int nclass,
                       float scale_in, int zi, float scale_out, int zo) {
    float* pvals = (float*)malloc(nclass * sizeof(float));
    for (volatile int b = 0; b < nbatch; b++) {
        int base = b * nclass;
        float t0 = -1e9f;  // max_val
        // 第一遍: 反量化并找最大值
        for (volatile int c = 0; c < nclass; c++) {
            float t1 = ((float)(*(pin + base + c)) - (float)zi) * scale_in;
            *(pvals + c) = t1;
            if (t1 > t0) t0 = t1;
        }
        // 第二遍: 计算exp并求和
        float t2 = 0.0f;  // sum
        for (volatile int c = 0; c < nclass; c++) {
            float t3 = expf(*(pvals + c) - t0);
            *(pvals + c) = t3;
            t2 += t3;
        }
        // 第三遍: 归一化并量化输出
        if (t2 == 0.0f) t2 = 1e-10f;
        for (volatile int c = 0; c < nclass; c++) {
            float t4 = *(pvals + c) / t2;
            int32_t t5 = (int32_t)roundf(t4 / scale_out) + zo;
            *(pout + base + c) = saturate_int8(t5);
        }
    }
    free(pvals);
}

// ============== 线性操作 ==============

// 全连接层: pout = pin @ pweight^T + pbias (Q16定点, 适配INT26)
// pscales_q16[o] = (si * pscales[o] / so) * (1<<16) 预计算数组
// 使用指针算术和手动偏移计算
static void op_fc(const int8_t* pin, int nbatch, int chn_in, int chn_out,
                  const int8_t* pweight, const int32_t* pbias, int8_t* pout,
                  int zi, const int32_t* pscales_q16, int zo) {
    // pscales_q16 已经是预计算的Q16定点缩放因子数组
    HW_BARRIER();
    
    // ===== 主计算循环 (纯int32_t运算, 适配INT26) =====
    for (volatile int b = 0; b < nbatch; b++) {
        int in_base = b * chn_in;
        int out_base = b * chn_out;
        for (volatile int o = 0; o < chn_out; o++) {
            int32_t t0 = 0;  // 累加器
            int w_base = o * chn_in;
            // MAC操作: 使用指针算术
            for (volatile int i = 0; i < chn_in; i++) {
                int32_t t1 = (int32_t)*(pin + in_base + i) - zi;
                int32_t t2 = (int32_t)*(pweight + w_base + i);
                t0 += t1 * t2;
            }
            // 加偏置
            if (pbias) t0 += *(pbias + o);
            // Q16量化输出: 分步计算适配INT26
            // (t0 >> 8) * scale >> 8 = (t0 * scale) >> 16
            int32_t t3 = ((t0 >> 8) * *(pscales_q16 + o)) >> 8;
            *(pout + out_base + o) = saturate_int8(t3 + zo);
        }
    }
}

// 批量矩阵乘法: pout[b] = pin1[b] @ pin2[b] (Q16定点)
// 维度: [nbatch, side_m, side_k] @ [nbatch, side_k, side_n] -> [nbatch, side_m, side_n]
// scale = (s1 * s2 / so) * (1<<16) 预计算定点乘数
static void op_batch_matmul(const int8_t* pin1, const int8_t* pin2, int8_t* pout,
                            int nbatch, int side_m, int side_k, int side_n,
                            int32_t scale, int z1, int z2, int zo) {
    // scale 已经是预计算的Q16定点乘数
    HW_BARRIER();
    
    // 预计算 stride (整数计算)
    int stride1_b = side_m * side_k;
    int stride2_b = side_k * side_n;
    int stride_out = side_m * side_n;
    
    // ===== 主计算循环 (纯int32_t运算) =====
    for (volatile int b = 0; b < nbatch; b++) {
        int base1 = b * stride1_b;
        int base2 = b * stride2_b;
        int base_out = b * stride_out;
        
        for (volatile int i = 0; i < side_m; i++) {
            for (volatile int j = 0; j < side_n; j++) {
                int32_t t0 = 0;  // 累加器
                // 内积计算
                for (volatile int l = 0; l < side_k; l++) {
                    int32_t t1 = (int32_t)*(pin1 + base1 + i * side_k + l) - z1;
                    int32_t t2 = (int32_t)*(pin2 + base2 + l * side_n + j) - z2;
                    t0 += t1 * t2;
                }
                // Q16量化: 分步计算 (t0 >> 8) * scale >> 8
                int32_t t3 = ((t0 >> 8) * scale) >> 8;
                *(pout + base_out + i * side_n + j) = saturate_int8(t3 + zo);
            }
        }
    }
}

// ============== 归约操作 ==============

// 均值: 沿指定维度求平均 (Q16定点)
// outer: 外层循环次数, reduce_len: 归约长度, inner: 内层循环次数
// scale = (si / so) * (1<<16) 预计算定点乘数
static void op_mean(const int8_t* pin, int8_t* pout,
                    int outer, int reduce_len, int inner,
                    int32_t scale, int zi, int zo) {
    int reduce_stride = inner;
    
    for (volatile int o = 0; o < outer; o++) {
        int in_base = o * reduce_len * inner;
        int out_base = o * inner;
        for (volatile int i = 0; i < inner; i++) {
            int32_t t0 = 0;  // 累加器
            // 沿归约维度求和
            for (volatile int r = 0; r < reduce_len; r++) {
                t0 += *(pin + in_base + r * reduce_stride + i);
            }
            // 计算均值: 使用位移 (若reduce_len为2的幂)
            int32_t t1;
            switch (reduce_len) {
                case 2:   t1 = t0 >> 1;  break;
                case 4:   t1 = t0 >> 2;  break;
                case 8:   t1 = t0 >> 3;  break;
                case 16:  t1 = t0 >> 4;  break;
                case 32:  t1 = t0 >> 5;  break;
                case 64:  t1 = t0 >> 6;  break;
                case 128: t1 = t0 >> 7;  break;
                case 256: t1 = t0 >> 8;  break;
                default:  t1 = t0 / reduce_len;  break;
            }
            // Q16量化输出: ((t1 - zi) * scale) >> 16
            int32_t t2 = t1 - zi;
            int32_t t3 = (t2 * scale) >> 16;
            *(pout + out_base + i) = saturate_int8(t3 + zo);
        }
    }
}

// ============== 形状操作 ==============

// 数据复制 (Reshape等)
static void op_copy(const int8_t* pin, int8_t* pout, int len) {
    if (pin != pout) memcpy(pout, pin, len);
}

// Transpose 3D: [d0, d1, d2] -> [perm[0], perm[1], perm[2]]
// 使用手动偏移计算
static void op_transpose_3d(const int8_t* pin, int8_t* pout,
                            int side0, int side1, int side2, 
                            int perm0, int perm1, int perm2) {
    int sides[3] = {side0, side1, side2};
    int perm[3] = {perm0, perm1, perm2};
    int new_sides[3] = {sides[perm[0]], sides[perm[1]], sides[perm[2]]};
    
    // 输入步长
    int stride_in1 = side2;
    int stride_in0 = side1 * side2;
    // 输出步长
    int stride_out1 = new_sides[2];
    int stride_out0 = new_sides[1] * new_sides[2];
    
    for (volatile int i0 = 0; i0 < side0; i0++) {
        for (volatile int i1 = 0; i1 < side1; i1++) {
            for (volatile int i2 = 0; i2 < side2; i2++) {
                // 输入偏移
                int in_off = i0 * stride_in0 + i1 * stride_in1 + i2;
                // 构建新索引
                int old_idx[3] = {i0, i1, i2};
                int new_idx[3];
                new_idx[0] = old_idx[perm[0]];
                new_idx[1] = old_idx[perm[1]];
                new_idx[2] = old_idx[perm[2]];
                // 输出偏移
                int out_off = new_idx[0] * stride_out0 + new_idx[1] * stride_out1 + new_idx[2];
                *(pout + out_off) = *(pin + in_off);
            }
        }
    }
}

// Transpose 4D: [d0, d1, d2, d3] -> [perm[0], perm[1], perm[2], perm[3]]
// 使用手动偏移计算和volatile循环变量
static void op_transpose_4d(const int8_t* pin, int8_t* pout,
                            int side0, int side1, int side2, int side3, 
                            int perm0, int perm1, int perm2, int perm3) {
    int sides[4] = {side0, side1, side2, side3};
    int perm[4] = {perm0, perm1, perm2, perm3};
    int new_sides[4] = {sides[perm[0]], sides[perm[1]], sides[perm[2]], sides[perm[3]]};
    
    // 输入步长计算
    int stride_in2 = side3;
    int stride_in1 = side2 * side3;
    int stride_in0 = side1 * side2 * side3;
    // 输出步长计算
    int stride_out2 = new_sides[3];
    int stride_out1 = new_sides[2] * new_sides[3];
    int stride_out0 = new_sides[1] * new_sides[2] * new_sides[3];
    
    for (volatile int i0 = 0; i0 < side0; i0++) {
        for (volatile int i1 = 0; i1 < side1; i1++) {
            for (volatile int i2 = 0; i2 < side2; i2++) {
                for (volatile int i3 = 0; i3 < side3; i3++) {
                    // 输入偏移: pin[i0, i1, i2, i3]
                    int in_off = i0 * stride_in0 + i1 * stride_in1 + i2 * stride_in2 + i3;
                    // 构建新索引
                    int old_idx[4] = {i0, i1, i2, i3};
                    int new_idx[4];
                    new_idx[0] = old_idx[perm[0]];
                    new_idx[1] = old_idx[perm[1]];
                    new_idx[2] = old_idx[perm[2]];
                    new_idx[3] = old_idx[perm[3]];
                    // 输出偏移: pout[new_idx[0], new_idx[1], new_idx[2], new_idx[3]]
                    int out_off = new_idx[0] * stride_out0 + new_idx[1] * stride_out1 + 
                                  new_idx[2] * stride_out2 + new_idx[3];
                    *(pout + out_off) = *(pin + in_off);
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
        """生成主程序源文件 - Bare-metal Hardware Verification Style"""
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
 * ECGformer INT8 主程序 - Bare-metal Hardware Verification Style
 * 自动生成 - 请勿手动修改
 * 
 * 编译: gcc -O3 -o ecgformer ecgformer_model.c -lm
 *       riscv64-unknown-elf-gcc -O3 -o ecgformer ecgformer_model.c -lm
 * 共享库: gcc -O3 -shared -fPIC -DBUILD_SHARED_LIB -o libecgformer.so ecgformer_model.c -lm
 * 
 * 命名约定:
 *   - 指针: p前缀 (pinput, poutput, ptensors)
 *   - 临时变量: t0, t1, t2... (模拟寄存器)
 *   - 无null检查, 直接指针算术
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

// ============== 张量存储 - Bare-metal Style ==============

// 激活张量内存池 (扁平分配)
static int8_t g_activation_pool[ACTIVATION_POOL_SIZE];

// 张量指针数组 (扁平int8_t*)
static int8_t* ptensors[NUM_TENSORS];

// 初始化张量指针 (无null检查, 直接赋值)
static void init_tensors(void) {
'''
        
        # 常量张量指向静态数组 (使用指针算术)
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            info = self.tensors_info[tid]
            dtype = info['dtype']
            if 'int8' in dtype:
                code += f'    *(ptensors + {tid}) = (int8_t*)weight_t{tid};\n'
            elif 'int32' in dtype:
                code += f'    *(ptensors + {tid}) = (int8_t*)bias_t{tid};\n'
        
        # 激活张量指向存储池 (使用指针算术)
        for tid in sorted(self.activation_tensors):
            if tid in offsets:
                code += f'    *(ptensors + {tid}) = g_activation_pool + {offsets[tid]};\n'
        
        code += '''    HW_BARRIER();  // 内存屏障确保初始化完成
}

// ============== 推理函数 ==============

int ecgformer_inference(const float* pinput_float, float* poutput_probs) {
    int8_t* pin = *(ptensors + ''' + str(input_tid) + ''');
    
    // 量化输入: 使用指针算术和volatile循环
    for (volatile int i = 0; i < INPUT_SIZE; i++) {
        float t0 = *(pinput_float + i);
        *(pin + i) = quantize_float(t0, INPUT_SCALE, INPUT_ZERO_POINT);
    }
    HW_BARRIER();
    
'''
        
        # 生成每个操作
        for op in self.ops:
            code += self._generate_op_code_modular(op)
        
        # 输出处理
        code += f'''
    // 反量化输出并找预测类别
    int8_t* pout = *(ptensors + {output_tid});
    int t0 = 0;  // pred
    float t1 = -1e9f;  // max_prob
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {{
        float t2 = dequantize_int8(*(pout + i), OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        *(poutput_probs + i) = t2;
        if (t2 > t1) {{
            t1 = t2;
            t0 = i;
        }}
    }}
    return t0;
}}

// 获取INT8输出 (用于硬件验证)
void ecgformer_get_int8_output(int8_t* poutput) {{
    int8_t* psrc = *(ptensors + {output_tid});
    // 使用memcpy替代循环
    memcpy(poutput, psrc, OUTPUT_CLASSES);
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

EXPORT int c_inference(const float* pinput, float* poutput) {{
    return ecgformer_inference(pinput, poutput);
}}

EXPORT void c_get_int8_output(int8_t* poutput) {{
    ecgformer_get_int8_output(poutput);
}}
#endif

// ============== 主函数 ==============

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {{
    init_tensors();
    
    printf("ECGformer INT8 Bare-metal C Implementation\\n");
    printf("==========================================\\n");
    
    // 测试用随机输入 (使用malloc, 无null检查)
    float* ptest_input = (float*)malloc(INPUT_SIZE << 2);  // INPUT_SIZE * 4
    for (volatile int i = 0; i < INPUT_SIZE; i++) {{
        *(ptest_input + i) = ((float)rand() / RAND_MAX - 0.5f);
    }}
    
    // 推理
    float* poutput_probs = (float*)malloc(OUTPUT_CLASSES << 2);  // OUTPUT_CLASSES * 4
    int t0 = ecgformer_inference(ptest_input, poutput_probs);
    
    printf("\\nPrediction Results:\\n");
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {{
        printf("  Class %d (%s): %.4f%s\\n", i, CLASS_NAMES[i], *(poutput_probs + i),
               i == t0 ? " <-- Predicted" : "");
    }}
    
    free(ptest_input);
    free(poutput_probs);
    return 0;
}}
#endif
'''
        
        path = os.path.join(output_dir, 'ecgformer_model.c')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_model.c")
    
    def _generate_op_code_modular(self, op):
        """为单个操作生成代码 - Q16 Fixed-Point Style (模块化版本)
        使用ptensors指针数组, 预计算Q16定点scale, 适配INT26中间结果
        """
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
            code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {in_size});\n'
        
        elif op_type == 'EXPAND_DIMS':
            in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
            code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {in_size});\n'
        
        elif op_type == 'TRANSPOSE':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            perm_tid = inputs[1]
            perm = self.weights_data.get(perm_tid, np.arange(len(in_shape))).flatten().tolist()
            
            if len(in_shape) == 3:
                code += f'    op_transpose_3d(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])});\n'
            elif len(in_shape) == 4:
                code += f'    op_transpose_4d(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])}, {int(perm[3])});\n'
            else:
                code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {out_size});\n'
        
        elif op_type == 'ADD':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            # Q16: scale1 = (s1/so) * (1<<16), scale2 = (s2/so) * (1<<16)
            scale1_q16 = int(round((s1 / out_scale) * (1 << 16)))
            scale2_q16 = int(round((s2 / out_scale) * (1 << 16)))
            code += f'    op_add(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'           {scale1_q16}, {z1}, {scale2_q16}, {z2}, {out_zp});\n'
        
        elif op_type == 'SUB':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(*(ptensors + {out_tid}), {out_zp}, {out_size});\n'
            else:
                scale1_q16 = int(round((s1 / out_scale) * (1 << 16)))
                scale2_q16 = int(round((s2 / out_scale) * (1 << 16)))
                code += f'    op_sub(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
                code += f'           {scale1_q16}, {z1}, {scale2_q16}, {z2}, {out_zp});\n'
        
        elif op_type == 'MUL':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            # Q16: scale = (s1 * s2 / so) * (1<<16)
            scale_q16 = int(round((s1 * s2 / out_scale) * (1 << 16)))
            code += f'    op_mul(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'           {scale_q16}, {z1}, {z2}, {out_zp});\n'
        
        elif op_type == 'SQUARED_DIFFERENCE':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(*(ptensors + {out_tid}), {out_zp}, {out_size});\n'
            else:
                # Q16: scale_diff = (s2/s1) * (1<<16), scale_out = (s1*s1/so) * (1<<16)
                scale_diff_q16 = int(round((s2 / s1) * (1 << 16)))
                scale_out_q16 = int(round((s1 * s1 / out_scale) * (1 << 16)))
                code += f'    op_squared_diff(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
                code += f'                    {scale_diff_q16}, {z1}, {z2}, {scale_out_q16}, {out_zp});\n'
        
        elif op_type == 'RSQRT':
            si, zi = self._get_scale_zp(inputs[0])
            # rsqrt使用浮点: scale_in = si, scale_out = so
            code += f'    op_rsqrt(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'             {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'FULLY_CONNECTED':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            chn_in = in_shape[-1] if len(in_shape) > 0 else 1
            nbatch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            chn_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            pbias_str = f'(const int32_t*)*(ptensors + {inputs[2]})' if has_bias else 'NULL'
            
            # 检查是否有预计算的 Q16 scales
            if op_id in self.q16_scales:
                # 使用预计算的 Q16 scale 数组
                pscales_str = f'pscales_q16_op{op_id}'
            else:
                # 单通道 scale: 预计算单个 Q16 值
                weight_info = self.tensors_info.get(weight_tid, {})
                weight_scales = weight_info.get('scales', np.array([]))
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                scale_q16 = int(round((si * single_scale / out_scale) * (1 << 16)))
                code += f'    {{ static const int32_t pws_q16[1] = {{{scale_q16}}};\n'
                pscales_str = 'pws_q16'
            
            code += f'    op_fc(*(ptensors + {inputs[0]}), {nbatch}, {chn_in}, {chn_out},\n'
            code += f'          (const int8_t*)*(ptensors + {weight_tid}), {pbias_str}, *(ptensors + {out_tid}),\n'
            code += f'          {zi}, {pscales_str}, {out_zp});\n'
            
            if op_id not in self.q16_scales:
                code += '    }\n'
        
        elif op_type == 'CONV_2D':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            chn_in = in_shape[-1] if len(in_shape) > 0 else 1
            spatial = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            chn_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            pbias_str = f'(const int32_t*)*(ptensors + {inputs[2]})' if has_bias else 'NULL'
            
            # 检查是否有预计算的 Q16 scales
            if op_id in self.q16_scales:
                pscales_str = f'pscales_q16_op{op_id}'
            else:
                weight_info = self.tensors_info.get(weight_tid, {})
                weight_scales = weight_info.get('scales', np.array([]))
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                scale_q16 = int(round((si * single_scale / out_scale) * (1 << 16)))
                code += f'    {{ static const int32_t pws_q16[1] = {{{scale_q16}}};\n'
                pscales_str = 'pws_q16'
            
            code += f'    op_fc(*(ptensors + {inputs[0]}), {spatial}, {chn_in}, {chn_out},\n'
            code += f'          (const int8_t*)*(ptensors + {weight_tid}), {pbias_str}, *(ptensors + {out_tid}),\n'
            code += f'          {zi}, {pscales_str}, {out_zp});\n'
            
            if op_id not in self.q16_scales:
                code += '    }\n'
        
        elif op_type == 'BATCH_MATMUL':
            in1_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            in2_shape = self.tensors_info.get(inputs[1], {}).get('shape', ())
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            
            nbatch = in1_shape[0] if len(in1_shape) > 2 else 1
            side_m = in1_shape[1] if len(in1_shape) > 1 else 1
            side_k = in1_shape[2] if len(in1_shape) > 2 else in1_shape[1] if len(in1_shape) > 1 else 1
            side_n = in2_shape[2] if len(in2_shape) > 2 else in2_shape[1] if len(in2_shape) > 1 else 1
            
            # Q16: scale = (s1 * s2 / so) * (1<<16)
            scale_q16 = int(round((s1 * s2 / out_scale) * (1 << 16)))
            code += f'    op_batch_matmul(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}),\n'
            code += f'                    {nbatch}, {side_m}, {side_k}, {side_n},\n'
            code += f'                    {scale_q16}, {z1}, {z2}, {out_zp});\n'
        
        elif op_type == 'MEAN':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            
            outer = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            reduce_len = in_shape[-1] if len(in_shape) > 0 else 1
            inner = 1
            
            # Q16: scale = (si / so) * (1<<16)
            scale_q16 = int(round((si / out_scale) * (1 << 16)))
            code += f'    op_mean(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}),\n'
            code += f'            {outer}, {reduce_len}, {inner},\n'
            code += f'            {scale_q16}, {zi}, {out_zp});\n'
        
        elif op_type == 'SOFTMAX':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            nbatch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            nclass = in_shape[-1] if len(in_shape) > 0 else 1
            
            # softmax使用浮点: scale_in = si, scale_out = so
            code += f'    op_softmax(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {nbatch}, {nclass},\n'
            code += f'               {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        else:
            code += f'    // TODO: Implement {op_type}\n'
        
        code += '\n'
        return code
    
    def _generate_header(self):
        """生成文件头 - Bare-metal Hardware Verification Style"""
        input_shape = tuple(self.input_details[0]['shape'])
        output_shape = tuple(self.output_details[0]['shape'])
        input_scale, input_zp = self._get_scale_zp(self.input_details[0]['index'])
        output_scale, output_zp = self._get_scale_zp(self.output_details[0]['index'])
        
        return f'''/**
 * ECGformer INT8 Bare-metal C Implementation
 * Auto-generated for Hardware Verification
 * 
 * Target: Custom RISC-V AI Accelerator
 * 
 * Build (GCC):
 *   gcc -O3 -o ecgformer ecgformer_standalone.c -lm
 * Build (RISC-V):
 *   riscv64-unknown-elf-gcc -O3 -o ecgformer ecgformer_standalone.c -lm
 * Shared Library:
 *   gcc -O3 -shared -fPIC -DBUILD_SHARED_LIB -o libecgformer.so ecgformer_standalone.c -lm
 * 
 * Coding Style:
 *   - Pointers: p prefix (pinput, poutput, ptensors)
 *   - Dimensions: side (inside, oside, fside)
 *   - Channels: chn_in, chn_out
 *   - Temps: t0, t1, t2... (mimic registers)
 *   - No structs: flat int8_t* pointers only
 *   - Bitwise ops: >> << instead of / * where possible
 *   - Volatile loops: for hardware interaction
 *   - Manual offset calculation: base + y*stride + x
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

static const char* CLASS_NAMES[5] = {{"N (Normal)", "S (SVEB)", "V (VEB)", "F (Fusion)", "Q (Unknown)"}};

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
        """生成张量存储 - Bare-metal Style (扁平指针数组)"""
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
        
        code = f'''// ============== 张量存储 - Bare-metal Style ==============
// 使用扁平int8_t*指针数组, 手动偏移计算

// 激活张量内存池 ({total_size} bytes)
static int8_t g_activation_pool[{total_size}];

// 张量指针数组 (扁平int8_t*)
static int8_t* ptensors[NUM_TENSORS];

// 初始化张量指针 (无null检查, 直接赋值)
static void init_tensors(void) {{
'''
        
        # 常量张量指向静态数组 (使用bare-metal命名)
        for tid in sorted(self.constant_tensors):
            if tid in self.weights_data:
                code += f'    *(ptensors + {tid}) = (int8_t*)const_t{tid};\n'
        
        # 激活张量指向存储池 (使用指针算术)
        for tid in sorted(self.activation_tensors):
            if tid in offsets:
                code += f'    *(ptensors + {tid}) = g_activation_pool + {offsets[tid]};\n'
        
        code += '''    HW_BARRIER();  // 确保初始化完成
}

'''
        return code
    
    def _generate_ops(self):
        """生成操作函数 - Bare-metal Hardware Verification Style"""
        return '''// ============== Bare-metal Hardware Verification Style ==============
// 命名约定: p前缀指针, side维度, chn通道, t0/t1临时变量
// 优化: 位移替代除法/乘法, volatile循环变量, 手动偏移计算

// 位移宏 (替代2的幂次除法/乘法)
#define DIV2(x)   ((x) >> 1)
#define DIV4(x)   ((x) >> 2)
#define DIV8(x)   ((x) >> 3)
#define DIV16(x)  ((x) >> 4)
#define DIV32(x)  ((x) >> 5)
#define DIV64(x)  ((x) >> 6)
#define DIV128(x) ((x) >> 7)
#define DIV256(x) ((x) >> 8)

#define MUL2(x)   ((x) << 1)
#define MUL4(x)   ((x) << 2)
#define MUL8(x)   ((x) << 3)
#define MUL16(x)  ((x) << 4)
#define MUL32(x)  ((x) << 5)
#define MUL64(x)  ((x) << 6)

// RISC-V 自定义指令宏
#define HW_BARRIER()  asm volatile("" ::: "memory")
#define HW_NOP()      asm volatile("nop")
#define HW_MAC_INIT(acc)      asm volatile(".insn r 0x0b, 0x0, 0x00, %0, x0, x0" : "=r"(acc))
#define HW_MAC_ACC(acc, a, b) asm volatile(".insn r 0x0b, 0x1, 0x00, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))
#define HW_QUANT(out, in, s)  asm volatile(".insn r 0x0b, 0x2, 0x00, %0, %1, %2" : "=r"(out) : "r"(in), "r"(s))

// ============== 操作实现 ==============
// Q16 定点格式: scale = (int32_t)((1.0 / float_scale) * (1 << 16))
// 量化公式: out = ((acc * scale) >> 16) + zp
// 适配INT26中间结果: 使用纯int32_t运算

static inline int8_t saturate_int8(int32_t t0) {
    if (t0 > 127) return 127;
    if (t0 < -128) return -128;
    return (int8_t)t0;
}

// Q16 定点量化: out = ((val >> 8) * scale >> 8) + zp (分步避免溢出)
static inline int8_t quantize_q16(int32_t t0, int32_t scale, int32_t zp) {
    int32_t t1 = ((t0 >> 8) * scale) >> 8;
    return saturate_int8(t1 + zp);
}

// 浮点输入量化 (仅用于外部接口，计算路径使用Q16)
static inline int8_t quantize_float(float t0, float scale, int32_t zp) {
    int32_t t1 = (int32_t)roundf(t0 / scale) + zp;
    return saturate_int8(t1);
}

// 反量化 (仅用于外部接口/调试)
static inline float dequantize_int8(int8_t t0, float scale, int32_t zp) {
    return ((float)t0 - (float)zp) * scale;
}

// 元素级加法: pout[i] = pin1[i] + pin2[i] (Q16定点)
// scale1 = (s1/so) * (1<<16), scale2 = (s2/so) * (1<<16)
static void op_add(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   int32_t scale1, int z1, int32_t scale2, int z2, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        int32_t t2 = (t0 * scale1 + t1 * scale2) >> 16;
        *(pout + i) = saturate_int8(t2 + zo);
    }
}

// 元素级减法: pout[i] = pin1[i] - pin2[i] (Q16定点)
static void op_sub(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   int32_t scale1, int z1, int32_t scale2, int z2, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        int32_t t2 = (t0 * scale1 - t1 * scale2) >> 16;
        *(pout + i) = saturate_int8(t2 + zo);
    }
}

// 元素级乘法: pout[i] = pin1[i] * pin2[i] (Q16定点)
// scale = (s1 * s2 / so) * (1<<16)
static void op_mul(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   int32_t scale, int z1, int z2, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        int32_t t2 = t0 * t1;
        int32_t t3 = ((t2 >> 8) * scale) >> 8;
        *(pout + i) = saturate_int8(t3 + zo);
    }
}

// 平方差: pout[i] = (pin1[i] - pin2[i])^2 (Q16定点)
// scale_diff = (s2/s1) * (1<<16), scale_out = (s1*s1/so) * (1<<16)
static void op_squared_diff(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                            int32_t scale_diff, int z1, int z2, int32_t scale_out, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin1 + i) - z1;
        int32_t t1 = (int32_t)*(pin2 + i) - z2;
        int32_t t2 = (t1 * scale_diff) >> 16;
        int32_t t3 = t0 - t2;
        int32_t t4 = ((t3 * t3) >> 8);
        int32_t t5 = (t4 * scale_out) >> 8;
        *(pout + i) = saturate_int8(t5 + zo);
    }
}

// 倒数平方根: pout[i] = 1/sqrt(pin[i])
// 使用浮点计算核心, 量化输入/输出
static void op_rsqrt(const int8_t* pin, int8_t* pout, int len,
                     float scale_in, int zi, float scale_out, int zo) {
    for (volatile int i = 0; i < len; i++) {
        int32_t t0 = (int32_t)*(pin + i) - zi;
        float t1 = (float)t0 * scale_in;
        if (t1 <= 0) t1 = 1e-10f;
        float t2 = 1.0f / sqrtf(t1);
        int32_t t3 = (int32_t)roundf(t2 / scale_out) + zo;
        *(pout + i) = saturate_int8(t3);
    }
}

// 全连接层: pout = pin @ pweight^T + pbias (Q16定点, 适配INT26)
// pscales_q16[o] = (si * pscales[o] / so) * (1<<16) 预计算数组
static void op_fc(const int8_t* pin, int nbatch, int chn_in, int chn_out,
                  const int8_t* pweight, const int32_t* pbias, int8_t* pout,
                  int zi, const int32_t* pscales_q16, int zo) {
    HW_BARRIER();
    
    // ===== 主计算循环 (纯int32_t运算) =====
    for (volatile int b = 0; b < nbatch; b++) {
        int in_base = b * chn_in;
        int out_base = b * chn_out;
        for (volatile int o = 0; o < chn_out; o++) {
            int32_t t0 = 0;  // 累加器
            int w_base = o * chn_in;
            for (volatile int i = 0; i < chn_in; i++) {
                int32_t t1 = (int32_t)*(pin + in_base + i) - zi;
                int32_t t2 = (int32_t)*(pweight + w_base + i);
                t0 += t1 * t2;
            }
            if (pbias) t0 += *(pbias + o);
            // Q16量化输出: (t0 >> 8) * scale >> 8, 分步避免溢出
            int32_t t3 = ((t0 >> 8) * *(pscales_q16 + o)) >> 8;
            *(pout + out_base + o) = saturate_int8(t3 + zo);
        }
    }
}

// 批量矩阵乘法: pout[b] = pin1[b] @ pin2[b] (Q16定点)
// scale = (s1 * s2 / so) * (1<<16) 预计算定点乘数
static void op_batch_matmul(const int8_t* pin1, const int8_t* pin2, int8_t* pout,
                            int nbatch, int side_m, int side_k, int side_n,
                            int32_t scale, int z1, int z2, int zo) {
    HW_BARRIER();
    
    // 预计算 stride (整数计算)
    int stride1_b = side_m * side_k;
    int stride2_b = side_k * side_n;
    int stride_out = side_m * side_n;
    
    // ===== 主计算循环 (纯int32_t运算) =====
    for (volatile int b = 0; b < nbatch; b++) {
        int base1 = b * stride1_b;
        int base2 = b * stride2_b;
        int base_out = b * stride_out;
        for (volatile int i = 0; i < side_m; i++) {
            for (volatile int j = 0; j < side_n; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < side_k; l++) {
                    int32_t t1 = (int32_t)*(pin1 + base1 + i * side_k + l) - z1;
                    int32_t t2 = (int32_t)*(pin2 + base2 + l * side_n + j) - z2;
                    t0 += t1 * t2;
                }
                // Q16量化: (t0 >> 8) * scale >> 8
                int32_t t3 = ((t0 >> 8) * scale) >> 8;
                *(pout + base_out + i * side_n + j) = saturate_int8(t3 + zo);
            }
        }
    }
}

// 均值: 沿指定维度求平均 (Q16定点)
// scale = (si / so) * (1<<16) 预计算定点乘数
static void op_mean(const int8_t* pin, int8_t* pout,
                    int outer, int reduce_len, int inner,
                    int32_t scale, int zi, int zo) {
    for (volatile int o = 0; o < outer; o++) {
        int in_base = o * reduce_len * inner;
        int out_base = o * inner;
        for (volatile int i = 0; i < inner; i++) {
            int32_t t0 = 0;
            for (volatile int r = 0; r < reduce_len; r++) {
                t0 += *(pin + in_base + r * inner + i);
            }
            // 使用位移替代常见2^n除法
            int32_t t1;
            switch (reduce_len) {
                case 2:   t1 = t0 >> 1;  break;
                case 4:   t1 = t0 >> 2;  break;
                case 8:   t1 = t0 >> 3;  break;
                case 16:  t1 = t0 >> 4;  break;
                case 32:  t1 = t0 >> 5;  break;
                case 64:  t1 = t0 >> 6;  break;
                case 128: t1 = t0 >> 7;  break;
                case 256: t1 = t0 >> 8;  break;
                default:  t1 = t0 / reduce_len;  break;
            }
            // Q16量化输出
            int32_t t2 = t1 - zi;
            int32_t t3 = (t2 * scale) >> 16;
            *(pout + out_base + i) = saturate_int8(t3 + zo);
        }
    }
}

// Softmax: 沿最后一个维度
// 使用浮点计算核心, 量化输入/输出
static void op_softmax(const int8_t* pin, int8_t* pout, int nbatch, int nclass,
                       float scale_in, int zi, float scale_out, int zo) {
    float* pvals = (float*)malloc(nclass * sizeof(float));
    for (volatile int b = 0; b < nbatch; b++) {
        int base = b * nclass;
        float t0 = -1e9f;  // max_val
        // 第一遍: 反量化并找最大值
        for (volatile int c = 0; c < nclass; c++) {
            float t1 = ((float)(*(pin + base + c)) - (float)zi) * scale_in;
            *(pvals + c) = t1;
            if (t1 > t0) t0 = t1;
        }
        // 第二遍: 计算exp并求和
        float t2 = 0.0f;
        for (volatile int c = 0; c < nclass; c++) {
            float t3 = expf(*(pvals + c) - t0);
            *(pvals + c) = t3;
            t2 += t3;
        }
        // 第三遍: 归一化并量化输出
        if (t2 == 0.0f) t2 = 1e-10f;
        for (volatile int c = 0; c < nclass; c++) {
            float t4 = *(pvals + c) / t2;
            int32_t t5 = (int32_t)roundf(t4 / scale_out) + zo;
            *(pout + base + c) = saturate_int8(t5);
        }
    }
    free(pvals);
}

// 数据复制 (Reshape)
static void op_copy(const int8_t* pin, int8_t* pout, int len) {
    if (pin != pout) memcpy(pout, pin, len);
}

// Transpose 3D: 使用手动偏移计算
static void op_transpose_3d(const int8_t* pin, int8_t* pout,
                            int side0, int side1, int side2, 
                            int perm0, int perm1, int perm2) {
    int sides[3] = {side0, side1, side2};
    int perm[3] = {perm0, perm1, perm2};
    int new_sides[3] = {sides[perm[0]], sides[perm[1]], sides[perm[2]]};
    
    int stride_in1 = side2;
    int stride_in0 = side1 * side2;
    int stride_out1 = new_sides[2];
    int stride_out0 = new_sides[1] * new_sides[2];
    
    for (volatile int i0 = 0; i0 < side0; i0++) {
        for (volatile int i1 = 0; i1 < side1; i1++) {
            for (volatile int i2 = 0; i2 < side2; i2++) {
                int in_off = i0 * stride_in0 + i1 * stride_in1 + i2;
                int old_idx[3] = {i0, i1, i2};
                int new_idx[3];
                new_idx[0] = old_idx[perm[0]];
                new_idx[1] = old_idx[perm[1]];
                new_idx[2] = old_idx[perm[2]];
                int out_off = new_idx[0] * stride_out0 + new_idx[1] * stride_out1 + new_idx[2];
                *(pout + out_off) = *(pin + in_off);
            }
        }
    }
}

// Transpose 4D: 使用手动偏移计算
static void op_transpose_4d(const int8_t* pin, int8_t* pout,
                            int side0, int side1, int side2, int side3, 
                            int perm0, int perm1, int perm2, int perm3) {
    int sides[4] = {side0, side1, side2, side3};
    int perm[4] = {perm0, perm1, perm2, perm3};
    int new_sides[4] = {sides[perm[0]], sides[perm[1]], sides[perm[2]], sides[perm[3]]};
    
    int stride_in2 = side3;
    int stride_in1 = side2 * side3;
    int stride_in0 = side1 * side2 * side3;
    int stride_out2 = new_sides[3];
    int stride_out1 = new_sides[2] * new_sides[3];
    int stride_out0 = new_sides[1] * new_sides[2] * new_sides[3];
    
    for (volatile int i0 = 0; i0 < side0; i0++) {
        for (volatile int i1 = 0; i1 < side1; i1++) {
            for (volatile int i2 = 0; i2 < side2; i2++) {
                for (volatile int i3 = 0; i3 < side3; i3++) {
                    int in_off = i0 * stride_in0 + i1 * stride_in1 + i2 * stride_in2 + i3;
                    int old_idx[4] = {i0, i1, i2, i3};
                    int new_idx[4];
                    new_idx[0] = old_idx[perm[0]];
                    new_idx[1] = old_idx[perm[1]];
                    new_idx[2] = old_idx[perm[2]];
                    new_idx[3] = old_idx[perm[3]];
                    int out_off = new_idx[0] * stride_out0 + new_idx[1] * stride_out1 + 
                                  new_idx[2] * stride_out2 + new_idx[3];
                    *(pout + out_off) = *(pin + in_off);
                }
            }
        }
    }
}

'''
    
    def _generate_inference(self):
        """生成推理函数 - Bare-metal Style"""
        input_tid = self.input_details[0]['index']
        output_tid = self.output_details[0]['index']
        input_scale, input_zp = self._get_scale_zp(input_tid)
        output_scale, output_zp = self._get_scale_zp(output_tid)
        
        code = '''// ============== 推理函数 - Bare-metal Style ==============

int ecgformer_inference(const float* pinput_float, float* poutput_probs) {
    int8_t* pin = *(ptensors + ''' + str(input_tid) + ''');
    
    // 量化输入: 使用指针算术和volatile循环
    for (volatile int i = 0; i < INPUT_SIZE; i++) {
        float t0 = *(pinput_float + i);
        *(pin + i) = quantize_float(t0, INPUT_SCALE, INPUT_ZERO_POINT);
    }
    HW_BARRIER();  // 确保输入写入完成
    
'''
        
        # 生成每个操作
        for op in self.ops:
            code += self._generate_op_code(op)
        
        # 输出处理
        code += f'''
    // 反量化输出并找预测类别
    int8_t* pout = *(ptensors + {output_tid});
    int t0 = 0;      // pred
    float t1 = -1e9f;  // max_prob
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {{
        float t2 = dequantize_int8(*(pout + i), OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        *(poutput_probs + i) = t2;
        if (t2 > t1) {{
            t1 = t2;
            t0 = i;
        }}
    }}
    return t0;
}}

// 获取INT8输出 (用于硬件验证)
void ecgformer_get_int8_output(int8_t* poutput) {{
    int8_t* psrc = *(ptensors + {output_tid});
    memcpy(poutput, psrc, OUTPUT_CLASSES);
}}

'''
        return code
    
    def _generate_op_code(self, op):
        """为单个操作生成代码 - Bare-metal Style (单文件版本)"""
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
            code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {in_size});\n'
        
        elif op_type == 'EXPAND_DIMS':
            in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
            code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {in_size});\n'
        
        elif op_type == 'TRANSPOSE':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            perm_tid = inputs[1]
            perm = self.weights_data.get(perm_tid, np.arange(len(in_shape))).flatten().tolist()
            
            if len(in_shape) == 3:
                code += f'    op_transpose_3d(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])});\n'
            elif len(in_shape) == 4:
                code += f'    op_transpose_4d(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])}, {int(perm[3])});\n'
            else:
                code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {out_size});\n'
        
        elif op_type == 'ADD':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            code += f'    op_add(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SUB':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(*(ptensors + {out_tid}), {out_zp}, {out_size});\n'
            else:
                code += f'    op_sub(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
                code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'MUL':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            code += f'    op_mul(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'           {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SQUARED_DIFFERENCE':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(*(ptensors + {out_tid}), {out_zp}, {out_size});\n'
            else:
                code += f'    op_squared_diff(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
                code += f'                    {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'RSQRT':
            si, zi = self._get_scale_zp(inputs[0])
            code += f'    op_rsqrt(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'             {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'FULLY_CONNECTED':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            chn_in = in_shape[-1] if len(in_shape) > 0 else 1
            nbatch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            chn_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            pbias_str = f'(const int32_t*)*(ptensors + {inputs[2]})' if has_bias else 'NULL'
            
            # 检查是否有per-channel scales
            weight_info = self.tensors_info.get(weight_tid, {})
            weight_scales = weight_info.get('scales', np.array([]))
            
            if len(weight_scales) > 1:
                pscales_str = f'scales_t{weight_tid}'
            else:
                # 使用malloc分配临时scale数组 (无null检查)
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                code += f'    {{ float* pws = (float*)malloc({chn_out} << 2);\n'
                code += f'      for(volatile int t0=0; t0<{chn_out}; t0++) *(pws+t0)={single_scale:.10e}f;\n'
                pscales_str = 'pws'
            
            code += f'    op_fc(*(ptensors + {inputs[0]}), {nbatch}, {chn_in}, {chn_out},\n'
            code += f'          (const int8_t*)*(ptensors + {weight_tid}), {pbias_str}, *(ptensors + {out_tid}),\n'
            code += f'          {si:.10e}f, {zi}, {pscales_str}, {out_scale:.10e}f, {out_zp});\n'
            
            if len(weight_scales) <= 1:
                code += '      free(pws); }\n'
        
        elif op_type == 'CONV_2D':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            chn_in = in_shape[-1] if len(in_shape) > 0 else 1
            spatial = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            chn_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            pbias_str = f'(const int32_t*)*(ptensors + {inputs[2]})' if has_bias else 'NULL'
            
            weight_info = self.tensors_info.get(weight_tid, {})
            weight_scales = weight_info.get('scales', np.array([]))
            
            if len(weight_scales) > 1:
                pscales_str = f'scales_t{weight_tid}'
            else:
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                code += f'    {{ float* pws = (float*)malloc({chn_out} << 2);\n'
                code += f'      for(volatile int t0=0; t0<{chn_out}; t0++) *(pws+t0)={single_scale:.10e}f;\n'
                pscales_str = 'pws'
            
            code += f'    op_fc(*(ptensors + {inputs[0]}), {spatial}, {chn_in}, {chn_out},\n'
            code += f'          (const int8_t*)*(ptensors + {weight_tid}), {pbias_str}, *(ptensors + {out_tid}),\n'
            code += f'          {si:.10e}f, {zi}, {pscales_str}, {out_scale:.10e}f, {out_zp});\n'
            
            if len(weight_scales) <= 1:
                code += '      free(pws); }\n'
        
        elif op_type == 'BATCH_MATMUL':
            in1_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            in2_shape = self.tensors_info.get(inputs[1], {}).get('shape', ())
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            
            nbatch = in1_shape[0] if len(in1_shape) > 2 else 1
            side_m = in1_shape[1] if len(in1_shape) > 1 else 1
            side_k = in1_shape[2] if len(in1_shape) > 2 else in1_shape[1] if len(in1_shape) > 1 else 1
            side_n = in2_shape[2] if len(in2_shape) > 2 else in2_shape[1] if len(in2_shape) > 1 else 1
            
            code += f'    op_batch_matmul(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}),\n'
            code += f'                    {nbatch}, {side_m}, {side_k}, {side_n},\n'
            code += f'                    {s1:.10e}f, {z1}, {s2:.10e}f, {z2}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'MEAN':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            
            # 假设沿最后一个轴
            outer = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            reduce_len = in_shape[-1] if len(in_shape) > 0 else 1
            inner = 1
            
            code += f'    op_mean(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}),\n'
            code += f'            {outer}, {reduce_len}, {inner},\n'
            code += f'            {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'SOFTMAX':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            # softmax沿最后一个维度，所以batch是前面所有维度的乘积
            nbatch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            nclass = in_shape[-1] if len(in_shape) > 0 else 1
            
            code += f'    op_softmax(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {nbatch}, {nclass},\n'
            code += f'               {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        else:
            code += f'    // TODO: Implement {op_type}\n'
        
        code += '\n'
        return code
    
    def _generate_main(self):
        """生成main函数和共享库接口 - Bare-metal Style"""
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

EXPORT int c_inference(const float* pinput, float* poutput) {
    return ecgformer_inference(pinput, poutput);
}

EXPORT void c_get_int8_output(int8_t* poutput) {
    ecgformer_get_int8_output(poutput);
}
#endif

// ============== 主函数 - Bare-metal Style ==============

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {
    init_tensors();
    
    printf("ECGformer INT8 Bare-metal C Implementation\\n");
    printf("==========================================\\n");
    
    // 测试用随机输入 (使用malloc, 无null检查)
    float* ptest_input = (float*)malloc(INPUT_SIZE << 2);  // INPUT_SIZE * 4
    for (volatile int i = 0; i < INPUT_SIZE; i++) {
        *(ptest_input + i) = ((float)rand() / RAND_MAX - 0.5f);
    }
    
    // 推理
    float* poutput_probs = (float*)malloc(OUTPUT_CLASSES << 2);  // OUTPUT_CLASSES * 4
    int t0 = ecgformer_inference(ptest_input, poutput_probs);
    
    printf("\\nPrediction Results:\\n");
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {
        printf("  Class %d (%s): %.4f%s\\n", i, CLASS_NAMES[i], *(poutput_probs + i),
               i == t0 ? " <-- Predicted" : "");
    }
    
    free(ptest_input);
    free(poutput_probs);
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
