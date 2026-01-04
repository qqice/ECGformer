/**
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
