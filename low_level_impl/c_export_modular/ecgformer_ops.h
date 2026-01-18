/**
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

// ============== 带转置输出的操作 (消除显式 transpose) ==============

// ADD + TRANSPOSE (0,2,1,3): 输入 [1,d1,d2,d3]，输出按 [1,d2,d1,d3] 布局写入
// 用于 Q/V 路径: [1,187,8,16] -> [1,8,187,16]
static void op_add_transpose_0213(const int8_t* pin1, const int8_t* pin2, int8_t* pout,
                                   int d1, int d2, int d3,
                                   int32_t scale1, int z1, int32_t scale2, int z2, int zo) {
    // 输入步长: [d1, d2, d3]
    int stride_in_d1 = d2 * d3;
    int stride_in_d2 = d3;
    // 输出步长: [d2, d1, d3] -> stride_d2=d1*d3, stride_d1=d3
    int stride_out_d2 = d1 * d3;
    int stride_out_d1 = d3;
    
    for (volatile int i1 = 0; i1 < d1; i1++) {
        for (volatile int i2 = 0; i2 < d2; i2++) {
            for (volatile int i3 = 0; i3 < d3; i3++) {
                int in_off = i1 * stride_in_d1 + i2 * stride_in_d2 + i3;
                int out_off = i2 * stride_out_d2 + i1 * stride_out_d1 + i3;
                int32_t t0 = (int32_t)*(pin1 + in_off) - z1;
                int32_t t1 = (int32_t)*(pin2 + in_off) - z2;
                int32_t t2 = (t0 * scale1 + t1 * scale2) >> 16;
                *(pout + out_off) = saturate_int8(t2 + zo);
            }
        }
    }
}

// MUL + TRANSPOSE (0,2,3,1): 输入 [1,d1,d2,d3]，输出按 [1,d2,d3,d1] 布局写入
// 用于 K 路径: [1,187,8,16] -> [1,8,16,187]
static void op_mul_transpose_0231(const int8_t* pin1, const int8_t* pin2, int8_t* pout,
                                   int d1, int d2, int d3,
                                   int32_t scale, int z1, int z2, int zo) {
    // 输入步长: [d1, d2, d3]
    int stride_in_d1 = d2 * d3;
    int stride_in_d2 = d3;
    // 输出步长: [d2, d3, d1] -> stride_d2=d3*d1, stride_d3=d1, stride_d1=1
    int stride_out_d2 = d3 * d1;
    int stride_out_d3 = d1;
    
    for (volatile int i1 = 0; i1 < d1; i1++) {
        for (volatile int i2 = 0; i2 < d2; i2++) {
            for (volatile int i3 = 0; i3 < d3; i3++) {
                int in_off = i1 * stride_in_d1 + i2 * stride_in_d2 + i3;
                int out_off = i2 * stride_out_d2 + i3 * stride_out_d3 + i1;
                int32_t t0 = (int32_t)*(pin1 + in_off) - z1;
                int32_t t1 = (int32_t)*(pin2 + in_off) - z2;
                int32_t t2 = t0 * t1;
                int32_t t3 = ((t2 >> 8) * scale) >> 8;
                *(pout + out_off) = saturate_int8(t3 + zo);
            }
        }
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

// 融合 FC 层: 从转置后的4D布局读取输入 (消除显式transpose)
// 输入布局: [1, d1, d2, d3] 需按 [1, d2, d1, d3] 逻辑访问 (perm = 0,2,1,3)
// 等效于: transpose(0,2,1,3) -> reshape -> FC
// 使用步长重映射避免实际数据拷贝
static void op_fc_from_transposed(const int8_t* pin, int d1, int d2, int d3,
                                  int chn_out,
                                  const int8_t* pweight, const int32_t* pbias, int8_t* pout,
                                  int zi, const int32_t* pscales_q16, int zo) {
    // 输入维度: [1, d1, d2, d3]，逻辑访问为 [1, d2, d1, d3]
    // 展平后: nbatch = d2, chn_in = d1 * d3
    HW_BARRIER();
    
    int chn_in = d1 * d3;  // 展平后的输入通道数
    int nbatch = d2;       // 批次数
    
    // 输入步长 (物理布局 [1, d1, d2, d3])
    int stride_d1 = d2 * d3;
    int stride_d2 = d3;
    
    for (volatile int b = 0; b < nbatch; b++) {
        int out_base = b * chn_out;
        for (volatile int o = 0; o < chn_out; o++) {
            int32_t t0 = 0;  // 累加器
            int w_base = o * chn_in;
            // 遍历 d1 和 d3，按转置后的顺序访问
            int in_idx = 0;
            for (volatile int i1 = 0; i1 < d1; i1++) {
                for (volatile int i3 = 0; i3 < d3; i3++) {
                    // 物理偏移: pin[0, i1, b, i3]
                    int in_off = i1 * stride_d1 + b * stride_d2 + i3;
                    int32_t t1 = (int32_t)*(pin + in_off) - zi;
                    int32_t t2 = (int32_t)*(pweight + w_base + in_idx);
                    t0 += t1 * t2;
                    in_idx++;
                }
            }
            if (pbias) t0 += *(pbias + o);
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

// ============== 乒乓缓冲区模式专用函数 ==============

#if CURRENT_MEMORY_MODE == MEMORY_MODE_PINGPONG

// 逐Head批量矩阵乘法 (用于256KB内存限制模式)
// 对于大型注意力矩阵，一次只处理一个head，避免内存溢出
// 输入维度: pin1 [num_heads, seq_len, head_dim], pin2 [num_heads, head_dim, seq_len] (或K^T)
// 输出维度: pout [num_heads, seq_len, seq_len] (但每次只计算一个head到临时缓冲)
// ptemp: 临时缓冲区，大小至少 seq_len * seq_len
static void op_batch_matmul_per_head(
    const int8_t* pin1, const int8_t* pin2, int8_t* pout, int8_t* ptemp,
    int num_heads, int seq_len, int head_dim, int side_n,
    int32_t scale, int z1, int z2, int zo) {
    
    HW_BARRIER();
    
    int stride1_h = seq_len * head_dim;  // Q/K 每个head的stride
    int stride2_h = head_dim * side_n;   // K^T/V 每个head的stride (side_n可能是seq_len或head_dim)
    int stride_out = seq_len * side_n;   // 输出每个head的stride
    
    // 逐Head处理，避免同时持有所有head的结果
    for (volatile int h = 0; h < num_heads; h++) {
        int base1 = h * stride1_h;
        int base2 = h * stride2_h;
        int base_out = h * stride_out;
        
        // 计算当前head的矩阵乘法结果到临时缓冲区
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < side_n; j++) {
                int32_t t0 = 0;  // 累加器
                for (volatile int l = 0; l < head_dim; l++) {
                    int32_t t1 = (int32_t)*(pin1 + base1 + i * head_dim + l) - z1;
                    int32_t t2 = (int32_t)*(pin2 + base2 + l * side_n + j) - z2;
                    t0 += t1 * t2;
                }
                int32_t t3 = ((t0 >> 8) * scale) >> 8;
                // 直接写入最终输出位置
                *(pout + base_out + i * side_n + j) = saturate_int8(t3 + zo);
            }
        }
        HW_BARRIER();  // 确保每个head计算完成
    }
}

// 逐Head注意力计算 (融合 Q*K^T + Softmax + *V)
// 这是内存高效的注意力实现，一次只处理一个head
// pQ: [num_heads, seq_len, head_dim]
// pK: [num_heads, seq_len, head_dim] (会被转置为 [num_heads, head_dim, seq_len])
// pV: [num_heads, seq_len, head_dim]
// pout: [num_heads, seq_len, head_dim]
// ptemp: 临时缓冲区，大小至少 seq_len * seq_len (用于单个head的注意力矩阵)
static void op_attention_per_head(
    const int8_t* pQ, const int8_t* pK, const int8_t* pV, int8_t* pout, int8_t* ptemp,
    int num_heads, int seq_len, int head_dim,
    int32_t scale_qk, int zq, int zk,      // Q*K^T 的量化参数
    float scale_softmax_in, int z_softmax,  // Softmax输入的量化参数
    int32_t scale_av, int za, int zv,       // Attn*V 的量化参数  
    float scale_softmax_out,                // Softmax输出的量化参数
    int zo) {
    
    HW_BARRIER();
    
    int stride_qkv = seq_len * head_dim;
    int stride_out = seq_len * head_dim;
    int attn_size = seq_len * seq_len;
    
    // 分配临时浮点缓冲区用于softmax
    float* pvals = (float*)malloc(seq_len * sizeof(float));
    
    // 逐Head处理
    for (volatile int h = 0; h < num_heads; h++) {
        int base_q = h * stride_qkv;
        int base_k = h * stride_qkv;
        int base_v = h * stride_qkv;
        int base_out = h * stride_out;
        
        // Step 1: 计算 Q * K^T -> ptemp [seq_len, seq_len]
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < seq_len; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < head_dim; l++) {
                    int32_t q_val = (int32_t)*(pQ + base_q + i * head_dim + l) - zq;
                    int32_t k_val = (int32_t)*(pK + base_k + j * head_dim + l) - zk;  // 注意K是按行读取，等效于K^T的列
                    t0 += q_val * k_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_qk) >> 8;
                *(ptemp + i * seq_len + j) = saturate_int8(t3);
            }
        }
        
        // Step 2: 对每一行应用Softmax (原地，使用浮点)
        for (volatile int i = 0; i < seq_len; i++) {
            int row_base = i * seq_len;
            
            // 找最大值
            float max_val = -1e9f;
            for (volatile int j = 0; j < seq_len; j++) {
                float val = ((float)(*(ptemp + row_base + j)) - (float)z_softmax) * scale_softmax_in;
                *(pvals + j) = val;  // 临时存储反量化值
                if (val > max_val) max_val = val;
            }
            
            // 计算exp并求和
            float sum = 0.0f;
            for (volatile int j = 0; j < seq_len; j++) {
                float exp_val = expf(*(pvals + j) - max_val);
                *(pvals + j) = exp_val;
                sum += exp_val;
            }
            
            // 归一化并写回
            if (sum == 0.0f) sum = 1e-10f;
            for (volatile int j = 0; j < seq_len; j++) {
                float prob = *(pvals + j) / sum;
                int32_t q_val = (int32_t)roundf(prob / scale_softmax_out) + z_softmax;
                *(ptemp + row_base + j) = saturate_int8(q_val);
            }
        }
        
        // Step 3: 计算 Attn * V -> pout [seq_len, head_dim]
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < head_dim; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < seq_len; l++) {
                    int32_t a_val = (int32_t)*(ptemp + i * seq_len + l) - za;
                    int32_t v_val = (int32_t)*(pV + base_v + l * head_dim + j) - zv;
                    t0 += a_val * v_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_av) >> 8;
                *(pout + base_out + i * head_dim + j) = saturate_int8(t3 + zo);
            }
        }
        
        HW_BARRIER();
    }
    
    free(pvals);
}

// 融合注意力层: Q @ K^T -> Transpose -> Softmax -> @ V
// 完全逐Head计算，只需要 seq_len * seq_len bytes 临时缓冲
// pQ: [1, num_heads, seq_len, head_dim]
// pK: [1, num_heads, head_dim, seq_len] (已转置)
// pV: [1, num_heads, seq_len, head_dim]
// pout: [1, num_heads, seq_len, head_dim] 或 [1, seq_len, num_heads, head_dim] (取决于transpose_out)
// transpose_out: 0=输出[1,heads,seq,dim], 1=输出[1,seq,heads,dim] (跳过后续transpose)
static void op_fused_attention_per_head(
    const int8_t* pQ, const int8_t* pK, const int8_t* pV, int8_t* pout,
    int num_heads, int seq_len, int head_dim, int transpose_out,
    int32_t scale_qk, int zq, int zk, int z_qk_out,  // Q*K^T 量化参数
    float scale_qk_out, float scale_softmax_out, int z_softmax,  // Softmax 量化参数
    int32_t scale_av, int zv, int zo) {  // Attn*V 量化参数
    
    HW_BARRIER();
    
    // 计算步长
    int stride_q = seq_len * head_dim;  // Q: (num_heads, seq_len, head_dim)
    int stride_k = head_dim * seq_len;  // K: (num_heads, head_dim, seq_len) - 已转置
    int stride_v = seq_len * head_dim;  // V: (num_heads, seq_len, head_dim)
    // 输出步长取决于是否转置输出
    int stride_out_h = transpose_out ? head_dim : (seq_len * head_dim);
    int stride_out_i = transpose_out ? (num_heads * head_dim) : head_dim;
    
    // 静态临时缓冲区 (避免malloc)
    // 使用全局静态缓冲区，大小足够存储一个head的注意力矩阵
    static int8_t s_attn_temp[187 * 187];  // seq_len * seq_len
    static float s_row_temp[187];  // seq_len，用于softmax行计算
    
    // 逐Head处理
    for (volatile int h = 0; h < num_heads; h++) {
        int base_q = h * stride_q;
        int base_k = h * stride_k;
        int base_v = h * stride_v;
        // 输出基址: transpose_out ? 无基址(逐元素计算) : h * (seq_len * head_dim)
        
        // Step 1: 计算 Q @ K^T -> attn_scores [seq_len, seq_len]
        // K的布局是 (num_heads, head_dim, seq_len)，相当于已经是K^T
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < seq_len; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < head_dim; l++) {
                    // Q[h, i, l] @ K[h, l, j]
                    int32_t q_val = (int32_t)*(pQ + base_q + i * head_dim + l) - zq;
                    int32_t k_val = (int32_t)*(pK + base_k + l * seq_len + j) - zk;
                    t0 += q_val * k_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_qk) >> 8;
                // 注意力分数存储在临时缓冲区，使用转置后的索引 (j, i)
                // 因为 TFLite 中的 TRANSPOSE 在 BATCH_MATMUL 之后
                s_attn_temp[j * seq_len + i] = saturate_int8(t3 + z_qk_out);
            }
        }
        
        // Step 2: 对每一行应用 Softmax (转置后的注意力矩阵)
        for (volatile int i = 0; i < seq_len; i++) {
            int row_base = i * seq_len;
            
            // 反量化并找最大值
            float max_val = -1e9f;
            for (volatile int j = 0; j < seq_len; j++) {
                float val = ((float)(s_attn_temp[row_base + j]) - (float)z_qk_out) * scale_qk_out;
                s_row_temp[j] = val;
                if (val > max_val) max_val = val;
            }
            
            // 计算 exp 并求和
            float sum = 0.0f;
            for (volatile int j = 0; j < seq_len; j++) {
                float exp_val = expf(s_row_temp[j] - max_val);
                s_row_temp[j] = exp_val;
                sum += exp_val;
            }
            
            // 归一化并量化写回
            if (sum < 1e-10f) sum = 1e-10f;
            for (volatile int j = 0; j < seq_len; j++) {
                float prob = s_row_temp[j] / sum;
                int32_t q_val = (int32_t)roundf(prob / scale_softmax_out) + z_softmax;
                s_attn_temp[row_base + j] = saturate_int8(q_val);
            }
        }
        
        // Step 3: 计算 Attn @ V -> output [seq_len, head_dim]
        // 输出布局取决于 transpose_out 参数:
        //   transpose_out=0: [num_heads, seq_len, head_dim] - 标准布局
        //   transpose_out=1: [seq_len, num_heads, head_dim] - 转置布局 (跳过后续transpose)
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < head_dim; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < seq_len; l++) {
                    // attn[i, l] @ V[h, l, j]
                    int32_t a_val = (int32_t)s_attn_temp[i * seq_len + l] - z_softmax;
                    int32_t v_val = (int32_t)*(pV + base_v + l * head_dim + j) - zv;
                    t0 += a_val * v_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_av) >> 8;
                // 根据 transpose_out 计算输出偏移
                // transpose_out=0: out[h, i, j] = h * (seq*dim) + i * dim + j
                // transpose_out=1: out[i, h, j] = i * (heads*dim) + h * dim + j
                int out_off = i * stride_out_i + h * stride_out_h + j;
                *(pout + out_off) = saturate_int8(t3 + zo);
            }
        }
        
        HW_BARRIER();
    }
}

#endif // CURRENT_MEMORY_MODE == MEMORY_MODE_PINGPONG

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
