/**
 * ECGformer INT8 模型配置
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_CONFIG_H
#define ECGFORMER_CONFIG_H

#include <stdint.h>

// ============== 模型配置 ==============

#define INPUT_SIZE 187
#define OUTPUT_CLASSES 5
#define NUM_TENSORS 262

// 内存管理模式
#define MEMORY_MODE_STATIC 0
#define MEMORY_MODE_REUSE 1
#define MEMORY_MODE_PINGPONG 2
#define CURRENT_MEMORY_MODE 2

// ============== 乒乓缓冲区模式 (严格256KB内存限制) ==============
// 策略: 槽位复用 + 融合注意力层 (逐Head计算)
// 融合后避免存储 (8, 187, 187) = 280KB 的注意力矩阵

#define MEMORY_LIMIT 262144
#define NUM_MEMORY_SLOTS 6

// 注意力计算参数
#define NUM_HEADS 8
#define SEQ_LEN 187
#define HEAD_DIM 16
#define ATTENTION_PER_HEAD (SEQ_LEN * SEQ_LEN)  // 34969 bytes per head

// 总内存池 (用于激活张量)
#define ACTIVATION_POOL_SIZE 155397

// 槽位大小数组
static const int g_slot_sizes[6] = {
    23936, 23936, 23936, 23936, 23936, 35717
};

// 槽位偏移数组 (预计算)
static const int g_slot_offsets[6] = {
    0, 23936, 47872, 71808, 95744, 119680
};

// 输入量化参数
#define INPUT_SCALE 3.5996969789e-02f
#define INPUT_ZERO_POINT 22

// 输出量化参数
#define OUTPUT_SCALE 3.9062500000e-03f
#define OUTPUT_ZERO_POINT -128

// 类别名称
static const char* CLASS_NAMES[5] = {"N (正常)", "S (室上性)", "V (室性)", "F (融合)", "Q (未知)"};

#endif // ECGFORMER_CONFIG_H
