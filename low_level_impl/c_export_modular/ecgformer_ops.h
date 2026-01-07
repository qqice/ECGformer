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
