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

// 内存管理模式: reuse (槽位复用)
#define MEMORY_MODE_REUSE 1

// 槽位复用模式: 每个槽位可被多个张量复用
#define NUM_MEMORY_SLOTS 5
#define ACTIVATION_POOL_SIZE 631312

// 槽位大小数组
static const int g_slot_sizes[5] = {
    23936, 279752, 23936, 23936, 279752
};

// 槽位偏移数组 (预计算)
static const int g_slot_offsets[5] = {
    0, 23936, 303688, 327624, 351560
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
