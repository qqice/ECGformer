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
#define ACTIVATION_POOL_SIZE 4911196

// 输入量化参数
#define INPUT_SCALE 3.5996969789e-02f
#define INPUT_ZERO_POINT 22

// 输出量化参数
#define OUTPUT_SCALE 3.9062500000e-03f
#define OUTPUT_ZERO_POINT -128

// 类别名称
static const char* CLASS_NAMES[5] = {"N (正常)", "S (室上性)", "V (室性)", "F (融合)", "Q (未知)"};

#endif // ECGFORMER_CONFIG_H
