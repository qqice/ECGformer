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

static inline int8_t saturate_int8(int32_t t0) {
    if (t0 > 127) return 127;
    if (t0 < -128) return -128;
    return (int8_t)t0;
}

static inline int8_t quantize_float(float t0, float scale, int32_t zp) {
    return saturate_int8((int32_t)roundf(t0 / scale) + zp);
}

static inline float dequantize_int8(int8_t t0, float scale, int32_t zp) {
    return ((float)t0 - (float)zp) * scale;
}

// ============== 元素级操作 ==============

// 元素级加法: pout[i] = pin1[i] + pin2[i] (量化)
static void op_add(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float t0 = s1 / so;
    float t1 = s2 / so;
    for (volatile int i = 0; i < len; i++) {
        float t2 = ((float)pin1[i] - z1) * t0 + ((float)pin2[i] - z2) * t1;
        pout[i] = saturate_int8((int32_t)roundf(t2) + zo);
    }
}

// 元素级减法: pout[i] = pin1[i] - pin2[i] (量化)
static void op_sub(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float t0 = s1 / so;
    float t1 = s2 / so;
    for (volatile int i = 0; i < len; i++) {
        float t2 = ((float)pin1[i] - z1) * t0 - ((float)pin2[i] - z2) * t1;
        pout[i] = saturate_int8((int32_t)roundf(t2) + zo);
    }
}

// 元素级乘法: pout[i] = pin1[i] * pin2[i] (量化)
static void op_mul(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                   float s1, int z1, float s2, int z2, float so, int zo) {
    float t0 = (s1 * s2) / so;
    for (volatile int i = 0; i < len; i++) {
        float t1 = ((float)pin1[i] - z1) * ((float)pin2[i] - z2) * t0;
        pout[i] = saturate_int8((int32_t)roundf(t1) + zo);
    }
}

// 平方差: pout[i] = (pin1[i] - pin2[i])^2
static void op_squared_diff(const int8_t* pin1, const int8_t* pin2, int8_t* pout, int len,
                            float s1, int z1, float s2, int z2, float so, int zo) {
    float t0 = (s1 * s1) / so;
    float t1 = s2 / s1;
    for (volatile int i = 0; i < len; i++) {
        float t2 = ((float)pin1[i] - z1) - ((float)pin2[i] - z2) * t1;
        float t3 = t2 * t2 * t0;
        pout[i] = saturate_int8((int32_t)roundf(t3) + zo);
    }
}

// ============== 激活函数 ==============

// 倒数平方根: pout[i] = 1/sqrt(pin[i])
static void op_rsqrt(const int8_t* pin, int8_t* pout, int len,
                     float si, int zi, float so, int zo) {
    for (volatile int i = 0; i < len; i++) {
        float t0 = ((float)pin[i] - zi) * si;
        float t1 = 1.0f / sqrtf(fmaxf(t0, 1e-12f));
        pout[i] = saturate_int8((int32_t)roundf(t1 / so) + zo);
    }
}

// Softmax: 沿最后一个维度, 使用指针算术
static void op_softmax(const int8_t* pin, int8_t* pout, int nbatch, int nclass,
                       float si, int zi, float so, int zo) {
    float* pvals = (float*)malloc(nclass << 2);  // nclass * sizeof(float) = nclass * 4
    for (volatile int b = 0; b < nbatch; b++) {
        int base = b * nclass;  // 手动偏移计算
        float t0 = -1e9f;  // max_val
        // 第一遍: 找最大值并反量化
        for (volatile int c = 0; c < nclass; c++) {
            pvals[c] = ((float)*(pin + base + c) - zi) * si;
            if (pvals[c] > t0) t0 = pvals[c];
        }
        // 第二遍: exp并求和
        float t1 = 0.0f;  // sum
        for (volatile int c = 0; c < nclass; c++) {
            pvals[c] = expf(pvals[c] - t0);
            t1 += pvals[c];
        }
        // 第三遍: 归一化并量化输出
        for (volatile int c = 0; c < nclass; c++) {
            float t2 = pvals[c] / t1;
            *(pout + base + c) = saturate_int8((int32_t)roundf(t2 / so) + zo);
        }
    }
    free(pvals);
}

// ============== 线性操作 ==============

// 全连接层: pout = pin @ pweight^T + pbias
// 优化: 预先计算每个输出通道的缩放因子数组，避免内层循环浮点除法
// 使用指针算术和手动偏移计算
static void op_fc(const int8_t* pin, int nbatch, int chn_in, int chn_out,
                  const int8_t* pweight, const int32_t* pbias, int8_t* pout,
                  float si, int zi, const float* pscales, float so, int zo) {
    // ===== 预先计算有效缩放因子数组 (在所有循环外) =====
    float* pscales_eff = (float*)malloc(chn_out * sizeof(float));
    float t0 = si / so;  // 公共因子
    for (volatile int o = 0; o < chn_out; o++) {
        *(pscales_eff + o) = t0 * *(pscales + o);  // 预计算: (si * pscales[o]) / so
    }
    HW_BARRIER();  // 确保预计算完成
    
    // ===== 主计算循环 (无浮点除法) =====
    for (volatile int b = 0; b < nbatch; b++) {
        int in_base = b * chn_in;      // 输入基址偏移
        int out_base = b * chn_out;    // 输出基址偏移
        for (volatile int o = 0; o < chn_out; o++) {
            int32_t t1 = 0;  // 累加器
            int w_base = o * chn_in;   // 权重行基址
            // MAC操作: 使用指针算术
            for (volatile int i = 0; i < chn_in; i++) {
                int32_t t2 = (int32_t)*(pin + in_base + i) - zi;
                int32_t t3 = (int32_t)*(pweight + w_base + i);
                t1 += t2 * t3;
            }
            // 加偏置
            if (pbias) t1 += *(pbias + o);
            // 量化输出: 使用预计算的缩放因子 (仅乘法，无除法)
            *(pout + out_base + o) = saturate_int8((int32_t)roundf(t1 * *(pscales_eff + o)) + zo);
        }
    }
    free(pscales_eff);
}

// 批量矩阵乘法: pout[b] = pin1[b] @ pin2[b]
// 维度: [nbatch, side_m, side_k] @ [nbatch, side_k, side_n] -> [nbatch, side_m, side_n]
// 优化: 浮点缩放转换为定点乘数+移位，彻底消除内层浮点运算
static void op_batch_matmul(const int8_t* pin1, const int8_t* pin2, int8_t* pout,
                            int nbatch, int side_m, int side_k, int side_n,
                            float s1, int z1, float s2, int z2, float so, int zo) {
    // ===== 预计算: 将浮点缩放转换为定点乘数+移位 =====
    // 原: t1 * ((s1 * s2) / so)
    // 优化: (t1 * multiplier) >> shift
    float t0 = (s1 * s2) / so;  // 有效缩放因子
    int shift = 15;  // 定点小数位数 (Q15格式)
    int32_t multiplier = (int32_t)roundf(t0 * (1 << shift));  // 定点乘数
    HW_BARRIER();  // 确保预计算完成
    
    // 预计算 stride (整数计算，无浮点)
    int stride1_b = side_m * side_k;  // pin1 batch stride
    int stride2_b = side_k * side_n;  // pin2 batch stride
    int stride_out = side_m * side_n; // pout batch stride
    
    // ===== 主计算循环 (纯整数运算) =====
    for (volatile int b = 0; b < nbatch; b++) {
        int base1 = b * stride1_b;
        int base2 = b * stride2_b;
        int base_out = b * stride_out;
        
        for (volatile int i = 0; i < side_m; i++) {
            for (volatile int j = 0; j < side_n; j++) {
                int32_t t1 = 0;  // 累加器
                // 内积计算
                for (volatile int l = 0; l < side_k; l++) {
                    // 手动计算偏移: pin1[b,i,l] = pin1[base1 + i*side_k + l]
                    //               pin2[b,l,j] = pin2[base2 + l*side_n + j]
                    int32_t t2 = (int32_t)*(pin1 + base1 + i * side_k + l) - z1;
                    int32_t t3 = (int32_t)*(pin2 + base2 + l * side_n + j) - z2;
                    t1 += t2 * t3;
                }
                // 定点量化: (t1 * multiplier) >> shift + zo
                // 使用64位中间结果防止溢出
                int64_t t4 = ((int64_t)t1 * (int64_t)multiplier) >> shift;
                *(pout + base_out + i * side_n + j) = saturate_int8((int32_t)t4 + zo);
            }
        }
    }
}

// ============== 归约操作 ==============

// 均值: 沿指定维度求平均
// outer: 外层循环次数, reduce_len: 归约长度, inner: 内层循环次数
static void op_mean(const int8_t* pin, int8_t* pout,
                    int outer, int reduce_len, int inner,
                    float si, int zi, float so, int zo) {
    float t0 = si / so;  // 缩放比
    int reduce_stride = inner;  // 归约维度步长
    
    for (volatile int o = 0; o < outer; o++) {
        int in_base = o * reduce_len * inner;
        int out_base = o * inner;
        for (volatile int i = 0; i < inner; i++) {
            int32_t t1 = 0;  // 累加器
            // 沿归约维度求和
            for (volatile int r = 0; r < reduce_len; r++) {
                t1 += *(pin + in_base + r * reduce_stride + i);
            }
            // 计算均值: 尝试使用位移 (若reduce_len为2的幂)
            int32_t t2;
            // 对常见2^n值使用位移
            switch (reduce_len) {
                case 2:   t2 = t1 >> 1;  break;
                case 4:   t2 = t1 >> 2;  break;
                case 8:   t2 = t1 >> 3;  break;
                case 16:  t2 = t1 >> 4;  break;
                case 32:  t2 = t1 >> 5;  break;
                case 64:  t2 = t1 >> 6;  break;
                case 128: t2 = t1 >> 7;  break;
                case 256: t2 = t1 >> 8;  break;
                default:  t2 = t1 / reduce_len;  break;
            }
            *(pout + out_base + i) = saturate_int8((int32_t)roundf((t2 - zi) * t0) + zo);
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
