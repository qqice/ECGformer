/**
 * ECGformer INT8 主程序 - Bare-metal Hardware Verification Style
 * 自动生成 - 请勿手动修改
 * 
 * 内存模式: reuse (槽位复用)
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

// ============== 张量存储 - 槽位复用模式 ==============

// 激活张量内存池
static int8_t g_activation_pool[ACTIVATION_POOL_SIZE];

// 张量指针数组 (扁平int8_t*)
static int8_t* ptensors[NUM_TENSORS];

// 初始化张量指针 (无null检查, 直接赋值)
static void init_tensors(void) {
    *(ptensors + 1) = (int8_t*)bias_t1;
    *(ptensors + 2) = (int8_t*)bias_t2;
    *(ptensors + 3) = (int8_t*)bias_t3;
    *(ptensors + 4) = (int8_t*)bias_t4;
    *(ptensors + 5) = (int8_t*)bias_t5;
    *(ptensors + 6) = (int8_t*)bias_t6;
    *(ptensors + 7) = (int8_t*)bias_t7;
    *(ptensors + 8) = (int8_t*)bias_t8;
    *(ptensors + 9) = (int8_t*)bias_t9;
    *(ptensors + 10) = (int8_t*)bias_t10;
    *(ptensors + 11) = (int8_t*)weight_t11;
    *(ptensors + 12) = (int8_t*)bias_t12;
    *(ptensors + 13) = (int8_t*)weight_t13;
    *(ptensors + 14) = (int8_t*)bias_t14;
    *(ptensors + 15) = (int8_t*)weight_t15;
    *(ptensors + 16) = (int8_t*)weight_t16;
    *(ptensors + 17) = (int8_t*)weight_t17;
    *(ptensors + 18) = (int8_t*)weight_t18;
    *(ptensors + 19) = (int8_t*)bias_t19;
    *(ptensors + 20) = (int8_t*)weight_t20;
    *(ptensors + 21) = (int8_t*)weight_t21;
    *(ptensors + 22) = (int8_t*)bias_t22;
    *(ptensors + 23) = (int8_t*)weight_t23;
    *(ptensors + 24) = (int8_t*)weight_t24;
    *(ptensors + 25) = (int8_t*)bias_t25;
    *(ptensors + 26) = (int8_t*)weight_t26;
    *(ptensors + 27) = (int8_t*)weight_t27;
    *(ptensors + 28) = (int8_t*)bias_t28;
    *(ptensors + 29) = (int8_t*)weight_t29;
    *(ptensors + 30) = (int8_t*)weight_t30;
    *(ptensors + 31) = (int8_t*)weight_t31;
    *(ptensors + 32) = (int8_t*)weight_t32;
    *(ptensors + 33) = (int8_t*)bias_t33;
    *(ptensors + 34) = (int8_t*)weight_t34;
    *(ptensors + 35) = (int8_t*)weight_t35;
    *(ptensors + 36) = (int8_t*)bias_t36;
    *(ptensors + 37) = (int8_t*)weight_t37;
    *(ptensors + 38) = (int8_t*)weight_t38;
    *(ptensors + 39) = (int8_t*)bias_t39;
    *(ptensors + 40) = (int8_t*)weight_t40;
    *(ptensors + 41) = (int8_t*)weight_t41;
    *(ptensors + 42) = (int8_t*)bias_t42;
    *(ptensors + 43) = (int8_t*)weight_t43;
    *(ptensors + 44) = (int8_t*)weight_t44;
    *(ptensors + 45) = (int8_t*)weight_t45;
    *(ptensors + 46) = (int8_t*)weight_t46;
    *(ptensors + 47) = (int8_t*)bias_t47;
    *(ptensors + 48) = (int8_t*)weight_t48;
    *(ptensors + 49) = (int8_t*)weight_t49;
    *(ptensors + 50) = (int8_t*)bias_t50;
    *(ptensors + 51) = (int8_t*)weight_t51;
    *(ptensors + 52) = (int8_t*)weight_t52;
    *(ptensors + 53) = (int8_t*)bias_t53;
    *(ptensors + 54) = (int8_t*)weight_t54;
    *(ptensors + 55) = (int8_t*)weight_t55;
    *(ptensors + 56) = (int8_t*)bias_t56;
    *(ptensors + 57) = (int8_t*)weight_t57;
    *(ptensors + 58) = (int8_t*)weight_t58;
    *(ptensors + 59) = (int8_t*)bias_t59;
    *(ptensors + 60) = (int8_t*)bias_t60;
    *(ptensors + 61) = (int8_t*)bias_t61;
    *(ptensors + 62) = (int8_t*)bias_t62;
    *(ptensors + 63) = (int8_t*)weight_t63;
    *(ptensors + 64) = (int8_t*)bias_t64;
    *(ptensors + 65) = (int8_t*)bias_t65;
    *(ptensors + 66) = (int8_t*)bias_t66;
    *(ptensors + 67) = (int8_t*)bias_t67;
    *(ptensors + 68) = (int8_t*)weight_t68;
    *(ptensors + 69) = (int8_t*)bias_t69;
    *(ptensors + 70) = (int8_t*)weight_t70;
    *(ptensors + 71) = (int8_t*)weight_t71;
    *(ptensors + 72) = (int8_t*)bias_t72;
    *(ptensors + 73) = (int8_t*)weight_t73;
    *(ptensors + 74) = (int8_t*)weight_t74;
    *(ptensors + 75) = (int8_t*)weight_t75;
    *(ptensors + 76) = (int8_t*)bias_t76;
    *(ptensors + 77) = (int8_t*)weight_t77;
    *(ptensors + 78) = (int8_t*)weight_t78;
    *(ptensors + 79) = (int8_t*)bias_t79;
    *(ptensors + 80) = (int8_t*)weight_t80;
    *(ptensors + 81) = (int8_t*)weight_t81;
    // 槽位复用模式: 输入张量预先设置
    *(ptensors + 0) = g_activation_pool + g_slot_offsets[0];
    HW_BARRIER();  // 内存屏障确保初始化完成
}

// ============== 推理函数 ==============

int ecgformer_inference(const float* pinput_float, float* poutput_probs) {
    int8_t* pin = *(ptensors + 0);
    
    // 量化输入: 使用指针算术和volatile循环
    for (volatile int i = 0; i < INPUT_SIZE; i++) {
        float t0 = *(pinput_float + i);
        *(pin + i) = quantize_float(t0, INPUT_SCALE, INPUT_ZERO_POINT);
    }
    HW_BARRIER();
    
    // Op#0: SQUARED_DIFFERENCE
    *(ptensors + 82) = g_activation_pool + g_slot_offsets[1];
    memset(*(ptensors + 82), -128, 187);

    // Op#1: ADD
    *(ptensors + 83) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 82), *(ptensors + 81), *(ptensors + 83), 187,
           65536, -128, 65536, -128, -128);

    // Op#2: SUB
    *(ptensors + 84) = g_activation_pool + g_slot_offsets[1];
    memset(*(ptensors + 84), 0, 187);

    // Op#3: RSQRT
    *(ptensors + 85) = g_activation_pool + g_slot_offsets[3];
    op_rsqrt(*(ptensors + 83), *(ptensors + 85), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#4: MUL
    *(ptensors + 86) = g_activation_pool + g_slot_offsets[2];
    op_mul(*(ptensors + 84), *(ptensors + 85), *(ptensors + 86), 187,
           8127, 0, -128, 0);

    // Op#5: FULLY_CONNECTED
    *(ptensors + 87) = g_activation_pool + g_slot_offsets[1];
    op_fc(*(ptensors + 86), 187, 1, 128,
          (const int8_t*)*(ptensors + 80), (const int32_t*)*(ptensors + 79), *(ptensors + 87),
          0, pscales_q16_op5, -1);

    // Op#6: RESHAPE
    *(ptensors + 88) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 87), *(ptensors + 88), 23936);

    // Op#7: ADD
    *(ptensors + 89) = g_activation_pool + g_slot_offsets[1];
    op_add(*(ptensors + 88), *(ptensors + 78), *(ptensors + 89), 23936,
           65537, -1, 75, 0, -1);

    // Op#8: TRANSPOSE
    *(ptensors + 90) = g_activation_pool + g_slot_offsets[3];
    op_transpose_4d(*(ptensors + 89), *(ptensors + 90), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#9: FULLY_CONNECTED
    *(ptensors + 91) = g_activation_pool + g_slot_offsets[1];
    op_fc(*(ptensors + 86), 187, 1, 128,
          (const int8_t*)*(ptensors + 77), (const int32_t*)*(ptensors + 76), *(ptensors + 91),
          0, pscales_q16_op9, 0);

    // Op#10: RESHAPE
    *(ptensors + 92) = g_activation_pool + g_slot_offsets[4];
    op_copy(*(ptensors + 91), *(ptensors + 92), 23936);

    // Op#11: ADD
    *(ptensors + 93) = g_activation_pool + g_slot_offsets[1];
    op_add(*(ptensors + 92), *(ptensors + 75), *(ptensors + 93), 23936,
           65769, 0, 1029, -6, -1);

    // Op#12: MUL
    *(ptensors + 94) = g_activation_pool + g_slot_offsets[4];
    op_mul(*(ptensors + 93), *(ptensors + 74), *(ptensors + 94), 23936,
           257, -1, -128, -1);

    // Op#13: TRANSPOSE
    *(ptensors + 95) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 94), *(ptensors + 95), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#14: BATCH_MATMUL
    *(ptensors + 96) = g_activation_pool + g_slot_offsets[4];
    op_batch_matmul(*(ptensors + 90), *(ptensors + 95), *(ptensors + 96),
                    1, 8, 187, 16,
                    23, -1, -1, -3);

    // Op#15: TRANSPOSE
    *(ptensors + 97) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 96), *(ptensors + 97), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#16: SOFTMAX
    *(ptensors + 98) = g_activation_pool + g_slot_offsets[4];
    op_softmax(*(ptensors + 97), *(ptensors + 98), 1496, 187,
               8.5386897553e-09f, -3, 3.9062500000e-03f, -128);

    // Op#17: FULLY_CONNECTED
    *(ptensors + 99) = g_activation_pool + g_slot_offsets[3];
    op_fc(*(ptensors + 86), 187, 1, 128,
          (const int8_t*)*(ptensors + 73), (const int32_t*)*(ptensors + 72), *(ptensors + 99),
          0, pscales_q16_op17, 2);

    // Op#18: RESHAPE
    *(ptensors + 100) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 99), *(ptensors + 100), 23936);

    // Op#19: ADD
    *(ptensors + 101) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 100), *(ptensors + 71), *(ptensors + 101), 23936,
           4037, 2, 63541, -14, -11);

    // Op#20: TRANSPOSE
    *(ptensors + 102) = g_activation_pool + g_slot_offsets[2];
    op_transpose_4d(*(ptensors + 101), *(ptensors + 102), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#21: BATCH_MATMUL
    *(ptensors + 103) = g_activation_pool + g_slot_offsets[3];
    op_batch_matmul(*(ptensors + 98), *(ptensors + 102), *(ptensors + 103),
                    1, 8, 187, 187,
                    256, -128, -11, -11);

    // Op#22: TRANSPOSE
    *(ptensors + 104) = g_activation_pool + g_slot_offsets[2];
    op_transpose_4d(*(ptensors + 103), *(ptensors + 104), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#23: RESHAPE
    *(ptensors + 105) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 104), *(ptensors + 105), 23936);

    // Op#24: FULLY_CONNECTED
    *(ptensors + 106) = g_activation_pool + g_slot_offsets[2];
    { static const int32_t pws_q16[1] = {0};
    op_fc(*(ptensors + 105), 187, 128, 1,
          (const int8_t*)*(ptensors + 70), (const int32_t*)*(ptensors + 69), *(ptensors + 106),
          -11, pws_q16, 22);
    }

    // Op#25: ADD
    *(ptensors + 107) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 106), *(ptensors + 0), *(ptensors + 107), 187,
           65536, 22, 65536, 22, 22);

    // Op#26: SQUARED_DIFFERENCE
    *(ptensors + 108) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 108), -128, 187);

    // Op#27: ADD
    *(ptensors + 109) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 108), *(ptensors + 81), *(ptensors + 109), 187,
           65536, -128, 65536, -128, -128);

    // Op#28: SUB
    *(ptensors + 110) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 110), 0, 187);

    // Op#29: RSQRT
    *(ptensors + 111) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 109), *(ptensors + 111), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#30: MUL
    *(ptensors + 112) = g_activation_pool + g_slot_offsets[2];
    op_mul(*(ptensors + 110), *(ptensors + 111), *(ptensors + 112), 187,
           8127, 0, -128, 0);

    // Op#31: EXPAND_DIMS
    *(ptensors + 113) = g_activation_pool + g_slot_offsets[0];
    op_copy(*(ptensors + 112), *(ptensors + 113), 187);

    // Op#32: CONV_2D
    *(ptensors + 114) = g_activation_pool + g_slot_offsets[2];
    op_fc(*(ptensors + 113), 187, 1, 4,
          (const int8_t*)*(ptensors + 68), (const int32_t*)*(ptensors + 64), *(ptensors + 114),
          0, pscales_q16_op32, 0);

    // Op#33: RESHAPE
    *(ptensors + 115) = g_activation_pool + g_slot_offsets[0];
    op_copy(*(ptensors + 114), *(ptensors + 115), 748);

    // Op#34: EXPAND_DIMS
    *(ptensors + 116) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 115), *(ptensors + 116), 748);

    // Op#35: CONV_2D
    *(ptensors + 117) = g_activation_pool + g_slot_offsets[0];
    { static const int32_t pws_q16[1] = {559};
    op_fc(*(ptensors + 116), 187, 4, 1,
          (const int8_t*)*(ptensors + 63), (const int32_t*)*(ptensors + 59), *(ptensors + 117),
          0, pws_q16, 0);
    }

    // Op#36: RESHAPE
    *(ptensors + 118) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 117), *(ptensors + 118), 187);

    // Op#37: ADD
    *(ptensors + 119) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 118), *(ptensors + 58), *(ptensors + 119), 187,
           0, 0, 636, -128, 20);

    // Op#38: ADD
    *(ptensors + 120) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 119), *(ptensors + 107), *(ptensors + 120), 187,
           65536, 20, 65536, 22, 20);

    // Op#39: SQUARED_DIFFERENCE
    *(ptensors + 121) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 121), -128, 187);

    // Op#40: ADD
    *(ptensors + 122) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 121), *(ptensors + 81), *(ptensors + 122), 187,
           65536, -128, 65536, -128, -128);

    // Op#41: SUB
    *(ptensors + 123) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 123), 0, 187);

    // Op#42: RSQRT
    *(ptensors + 124) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 122), *(ptensors + 124), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#43: MUL
    *(ptensors + 125) = g_activation_pool + g_slot_offsets[3];
    op_mul(*(ptensors + 123), *(ptensors + 124), *(ptensors + 125), 187,
           8127, 0, -128, 0);

    // Op#44: FULLY_CONNECTED
    *(ptensors + 126) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 125), 187, 1, 128,
          (const int8_t*)*(ptensors + 57), (const int32_t*)*(ptensors + 56), *(ptensors + 126),
          0, pscales_q16_op44, -1);

    // Op#45: RESHAPE
    *(ptensors + 127) = g_activation_pool + g_slot_offsets[1];
    op_copy(*(ptensors + 126), *(ptensors + 127), 23936);

    // Op#46: ADD
    *(ptensors + 128) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 127), *(ptensors + 55), *(ptensors + 128), 23936,
           65536, -1, 14, 0, -1);

    // Op#47: TRANSPOSE
    *(ptensors + 129) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 128), *(ptensors + 129), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#48: FULLY_CONNECTED
    *(ptensors + 130) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 125), 187, 1, 128,
          (const int8_t*)*(ptensors + 54), (const int32_t*)*(ptensors + 53), *(ptensors + 130),
          0, pscales_q16_op48, 1);

    // Op#49: RESHAPE
    *(ptensors + 131) = g_activation_pool + g_slot_offsets[4];
    op_copy(*(ptensors + 130), *(ptensors + 131), 23936);

    // Op#50: ADD
    *(ptensors + 132) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 131), *(ptensors + 52), *(ptensors + 132), 23936,
           65694, 1, 497, -6, 1);

    // Op#51: MUL
    *(ptensors + 133) = g_activation_pool + g_slot_offsets[4];
    op_mul(*(ptensors + 132), *(ptensors + 74), *(ptensors + 133), 23936,
           257, 1, -128, 1);

    // Op#52: TRANSPOSE
    *(ptensors + 134) = g_activation_pool + g_slot_offsets[0];
    op_transpose_4d(*(ptensors + 133), *(ptensors + 134), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#53: BATCH_MATMUL
    *(ptensors + 135) = g_activation_pool + g_slot_offsets[4];
    op_batch_matmul(*(ptensors + 129), *(ptensors + 134), *(ptensors + 135),
                    1, 8, 187, 16,
                    253, -1, 1, 29);

    // Op#54: TRANSPOSE
    *(ptensors + 136) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 135), *(ptensors + 136), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#55: SOFTMAX
    *(ptensors + 137) = g_activation_pool + g_slot_offsets[4];
    op_softmax(*(ptensors + 136), *(ptensors + 137), 1496, 187,
               2.1187917199e-08f, 29, 3.9062500000e-03f, -128);

    // Op#56: FULLY_CONNECTED
    *(ptensors + 138) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 125), 187, 1, 128,
          (const int8_t*)*(ptensors + 51), (const int32_t*)*(ptensors + 50), *(ptensors + 138),
          0, pscales_q16_op56, 0);

    // Op#57: RESHAPE
    *(ptensors + 139) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 138), *(ptensors + 139), 23936);

    // Op#58: ADD
    *(ptensors + 140) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 139), *(ptensors + 49), *(ptensors + 140), 23936,
           33911, 0, 85324, -6, 6);

    // Op#59: TRANSPOSE
    *(ptensors + 141) = g_activation_pool + g_slot_offsets[3];
    op_transpose_4d(*(ptensors + 140), *(ptensors + 141), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#60: BATCH_MATMUL
    *(ptensors + 142) = g_activation_pool + g_slot_offsets[0];
    op_batch_matmul(*(ptensors + 137), *(ptensors + 141), *(ptensors + 142),
                    1, 8, 187, 187,
                    256, -128, 6, 6);

    // Op#61: TRANSPOSE
    *(ptensors + 143) = g_activation_pool + g_slot_offsets[3];
    op_transpose_4d(*(ptensors + 142), *(ptensors + 143), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#62: RESHAPE
    *(ptensors + 144) = g_activation_pool + g_slot_offsets[0];
    op_copy(*(ptensors + 143), *(ptensors + 144), 23936);

    // Op#63: FULLY_CONNECTED
    *(ptensors + 145) = g_activation_pool + g_slot_offsets[3];
    { static const int32_t pws_q16[1] = {0};
    op_fc(*(ptensors + 144), 187, 128, 1,
          (const int8_t*)*(ptensors + 48), (const int32_t*)*(ptensors + 47), *(ptensors + 145),
          6, pws_q16, 20);
    }

    // Op#64: ADD
    *(ptensors + 146) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 145), *(ptensors + 120), *(ptensors + 146), 187,
           65536, 20, 65536, 20, 20);

    // Op#65: SQUARED_DIFFERENCE
    *(ptensors + 147) = g_activation_pool + g_slot_offsets[2];
    memset(*(ptensors + 147), -128, 187);

    // Op#66: ADD
    *(ptensors + 148) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 147), *(ptensors + 81), *(ptensors + 148), 187,
           65536, -128, 65536, -128, -128);

    // Op#67: SUB
    *(ptensors + 149) = g_activation_pool + g_slot_offsets[2];
    memset(*(ptensors + 149), 0, 187);

    // Op#68: RSQRT
    *(ptensors + 150) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 148), *(ptensors + 150), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#69: MUL
    *(ptensors + 151) = g_activation_pool + g_slot_offsets[3];
    op_mul(*(ptensors + 149), *(ptensors + 150), *(ptensors + 151), 187,
           8127, 0, -128, 0);

    // Op#70: EXPAND_DIMS
    *(ptensors + 152) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 151), *(ptensors + 152), 187);

    // Op#71: CONV_2D
    *(ptensors + 153) = g_activation_pool + g_slot_offsets[3];
    op_fc(*(ptensors + 152), 187, 1, 4,
          (const int8_t*)*(ptensors + 46), (const int32_t*)*(ptensors + 65), *(ptensors + 153),
          0, pscales_q16_op71, 0);

    // Op#72: RESHAPE
    *(ptensors + 154) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 153), *(ptensors + 154), 748);

    // Op#73: EXPAND_DIMS
    *(ptensors + 155) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 154), *(ptensors + 155), 748);

    // Op#74: CONV_2D
    *(ptensors + 156) = g_activation_pool + g_slot_offsets[2];
    { static const int32_t pws_q16[1] = {289};
    op_fc(*(ptensors + 155), 187, 4, 1,
          (const int8_t*)*(ptensors + 45), (const int32_t*)*(ptensors + 60), *(ptensors + 156),
          0, pws_q16, 0);
    }

    // Op#75: RESHAPE
    *(ptensors + 157) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 156), *(ptensors + 157), 187);

    // Op#76: ADD
    *(ptensors + 158) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 157), *(ptensors + 44), *(ptensors + 158), 187,
           0, 0, 636, -128, 17);

    // Op#77: ADD
    *(ptensors + 159) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 158), *(ptensors + 146), *(ptensors + 159), 187,
           65536, 17, 65536, 20, 17);

    // Op#78: SQUARED_DIFFERENCE
    *(ptensors + 160) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 160), -128, 187);

    // Op#79: ADD
    *(ptensors + 161) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 160), *(ptensors + 81), *(ptensors + 161), 187,
           65536, -128, 65536, -128, -128);

    // Op#80: SUB
    *(ptensors + 162) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 162), 0, 187);

    // Op#81: RSQRT
    *(ptensors + 163) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 161), *(ptensors + 163), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#82: MUL
    *(ptensors + 164) = g_activation_pool + g_slot_offsets[2];
    op_mul(*(ptensors + 162), *(ptensors + 163), *(ptensors + 164), 187,
           8127, 0, -128, 0);

    // Op#83: FULLY_CONNECTED
    *(ptensors + 165) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 164), 187, 1, 128,
          (const int8_t*)*(ptensors + 43), (const int32_t*)*(ptensors + 42), *(ptensors + 165),
          0, pscales_q16_op83, -1);

    // Op#84: RESHAPE
    *(ptensors + 166) = g_activation_pool + g_slot_offsets[1];
    op_copy(*(ptensors + 165), *(ptensors + 166), 23936);

    // Op#85: ADD
    *(ptensors + 167) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 166), *(ptensors + 41), *(ptensors + 167), 23936,
           65536, -1, 30, 0, -1);

    // Op#86: TRANSPOSE
    *(ptensors + 168) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 167), *(ptensors + 168), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#87: FULLY_CONNECTED
    *(ptensors + 169) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 164), 187, 1, 128,
          (const int8_t*)*(ptensors + 40), (const int32_t*)*(ptensors + 39), *(ptensors + 169),
          0, pscales_q16_op87, -2);

    // Op#88: RESHAPE
    *(ptensors + 170) = g_activation_pool + g_slot_offsets[4];
    op_copy(*(ptensors + 169), *(ptensors + 170), 23936);

    // Op#89: ADD
    *(ptensors + 171) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 170), *(ptensors + 38), *(ptensors + 171), 23936,
           65534, -2, 635, -1, -2);

    // Op#90: MUL
    *(ptensors + 172) = g_activation_pool + g_slot_offsets[4];
    op_mul(*(ptensors + 171), *(ptensors + 74), *(ptensors + 172), 23936,
           257, -2, -128, -2);

    // Op#91: TRANSPOSE
    *(ptensors + 173) = g_activation_pool + g_slot_offsets[0];
    op_transpose_4d(*(ptensors + 172), *(ptensors + 173), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#92: BATCH_MATMUL
    *(ptensors + 174) = g_activation_pool + g_slot_offsets[4];
    op_batch_matmul(*(ptensors + 168), *(ptensors + 173), *(ptensors + 174),
                    1, 8, 187, 16,
                    294, -1, -2, -73);

    // Op#93: TRANSPOSE
    *(ptensors + 175) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 174), *(ptensors + 175), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#94: SOFTMAX
    *(ptensors + 176) = g_activation_pool + g_slot_offsets[4];
    op_softmax(*(ptensors + 175), *(ptensors + 176), 1496, 187,
               4.2088466046e-09f, -73, 3.9062500000e-03f, -128);

    // Op#95: FULLY_CONNECTED
    *(ptensors + 177) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 164), 187, 1, 128,
          (const int8_t*)*(ptensors + 37), (const int32_t*)*(ptensors + 36), *(ptensors + 177),
          0, pscales_q16_op95, -2);

    // Op#96: RESHAPE
    *(ptensors + 178) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 177), *(ptensors + 178), 23936);

    // Op#97: ADD
    *(ptensors + 179) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 178), *(ptensors + 35), *(ptensors + 179), 23936,
           13318, -2, 62684, -8, 0);

    // Op#98: TRANSPOSE
    *(ptensors + 180) = g_activation_pool + g_slot_offsets[2];
    op_transpose_4d(*(ptensors + 179), *(ptensors + 180), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#99: BATCH_MATMUL
    *(ptensors + 181) = g_activation_pool + g_slot_offsets[0];
    op_batch_matmul(*(ptensors + 176), *(ptensors + 180), *(ptensors + 181),
                    1, 8, 187, 187,
                    256, -128, 0, 0);

    // Op#100: TRANSPOSE
    *(ptensors + 182) = g_activation_pool + g_slot_offsets[2];
    op_transpose_4d(*(ptensors + 181), *(ptensors + 182), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#101: RESHAPE
    *(ptensors + 183) = g_activation_pool + g_slot_offsets[0];
    op_copy(*(ptensors + 182), *(ptensors + 183), 23936);

    // Op#102: FULLY_CONNECTED
    *(ptensors + 184) = g_activation_pool + g_slot_offsets[2];
    { static const int32_t pws_q16[1] = {0};
    op_fc(*(ptensors + 183), 187, 128, 1,
          (const int8_t*)*(ptensors + 34), (const int32_t*)*(ptensors + 33), *(ptensors + 184),
          0, pws_q16, 17);
    }

    // Op#103: ADD
    *(ptensors + 185) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 184), *(ptensors + 159), *(ptensors + 185), 187,
           65536, 17, 65536, 17, 17);

    // Op#104: SQUARED_DIFFERENCE
    *(ptensors + 186) = g_activation_pool + g_slot_offsets[2];
    memset(*(ptensors + 186), -128, 187);

    // Op#105: ADD
    *(ptensors + 187) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 186), *(ptensors + 81), *(ptensors + 187), 187,
           65536, -128, 65536, -128, -128);

    // Op#106: SUB
    *(ptensors + 188) = g_activation_pool + g_slot_offsets[2];
    memset(*(ptensors + 188), 0, 187);

    // Op#107: RSQRT
    *(ptensors + 189) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 187), *(ptensors + 189), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#108: MUL
    *(ptensors + 190) = g_activation_pool + g_slot_offsets[3];
    op_mul(*(ptensors + 188), *(ptensors + 189), *(ptensors + 190), 187,
           8127, 0, -128, 0);

    // Op#109: EXPAND_DIMS
    *(ptensors + 191) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 190), *(ptensors + 191), 187);

    // Op#110: CONV_2D
    *(ptensors + 192) = g_activation_pool + g_slot_offsets[3];
    op_fc(*(ptensors + 191), 187, 1, 4,
          (const int8_t*)*(ptensors + 32), (const int32_t*)*(ptensors + 66), *(ptensors + 192),
          0, pscales_q16_op110, 0);

    // Op#111: RESHAPE
    *(ptensors + 193) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 192), *(ptensors + 193), 748);

    // Op#112: EXPAND_DIMS
    *(ptensors + 194) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 193), *(ptensors + 194), 748);

    // Op#113: CONV_2D
    *(ptensors + 195) = g_activation_pool + g_slot_offsets[2];
    { static const int32_t pws_q16[1] = {222};
    op_fc(*(ptensors + 194), 187, 4, 1,
          (const int8_t*)*(ptensors + 31), (const int32_t*)*(ptensors + 61), *(ptensors + 195),
          0, pws_q16, 0);
    }

    // Op#114: RESHAPE
    *(ptensors + 196) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 195), *(ptensors + 196), 187);

    // Op#115: ADD
    *(ptensors + 197) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 196), *(ptensors + 30), *(ptensors + 197), 187,
           0, 0, 636, -128, 15);

    // Op#116: ADD
    *(ptensors + 198) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 197), *(ptensors + 185), *(ptensors + 198), 187,
           65536, 15, 65536, 17, 15);

    // Op#117: SQUARED_DIFFERENCE
    *(ptensors + 199) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 199), -128, 187);

    // Op#118: ADD
    *(ptensors + 200) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 199), *(ptensors + 81), *(ptensors + 200), 187,
           65536, -128, 65536, -128, -128);

    // Op#119: SUB
    *(ptensors + 201) = g_activation_pool + g_slot_offsets[0];
    memset(*(ptensors + 201), 0, 187);

    // Op#120: RSQRT
    *(ptensors + 202) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 200), *(ptensors + 202), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#121: MUL
    *(ptensors + 203) = g_activation_pool + g_slot_offsets[2];
    op_mul(*(ptensors + 201), *(ptensors + 202), *(ptensors + 203), 187,
           8127, 0, -128, 0);

    // Op#122: FULLY_CONNECTED
    *(ptensors + 204) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 203), 187, 1, 128,
          (const int8_t*)*(ptensors + 29), (const int32_t*)*(ptensors + 28), *(ptensors + 204),
          0, pscales_q16_op122, -2);

    // Op#123: RESHAPE
    *(ptensors + 205) = g_activation_pool + g_slot_offsets[1];
    op_copy(*(ptensors + 204), *(ptensors + 205), 23936);

    // Op#124: ADD
    *(ptensors + 206) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 205), *(ptensors + 27), *(ptensors + 206), 23936,
           65537, -2, 568, 0, -2);

    // Op#125: TRANSPOSE
    *(ptensors + 207) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 206), *(ptensors + 207), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#126: FULLY_CONNECTED
    *(ptensors + 208) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 203), 187, 1, 128,
          (const int8_t*)*(ptensors + 26), (const int32_t*)*(ptensors + 25), *(ptensors + 208),
          0, pscales_q16_op126, -1);

    // Op#127: RESHAPE
    *(ptensors + 209) = g_activation_pool + g_slot_offsets[4];
    op_copy(*(ptensors + 208), *(ptensors + 209), 23936);

    // Op#128: ADD
    *(ptensors + 210) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 209), *(ptensors + 24), *(ptensors + 210), 23936,
           61645, -1, 17256, 5, 1);

    // Op#129: MUL
    *(ptensors + 211) = g_activation_pool + g_slot_offsets[4];
    op_mul(*(ptensors + 210), *(ptensors + 74), *(ptensors + 211), 23936,
           257, 1, -128, 1);

    // Op#130: TRANSPOSE
    *(ptensors + 212) = g_activation_pool + g_slot_offsets[0];
    op_transpose_4d(*(ptensors + 211), *(ptensors + 212), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#131: BATCH_MATMUL
    *(ptensors + 213) = g_activation_pool + g_slot_offsets[4];
    op_batch_matmul(*(ptensors + 207), *(ptensors + 212), *(ptensors + 213),
                    1, 8, 187, 16,
                    0, -2, 1, 0);

    // Op#132: TRANSPOSE
    *(ptensors + 214) = g_activation_pool + g_slot_offsets[1];
    op_transpose_4d(*(ptensors + 213), *(ptensors + 214), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#133: SOFTMAX
    *(ptensors + 215) = g_activation_pool + g_slot_offsets[4];
    op_softmax(*(ptensors + 214), *(ptensors + 215), 1496, 187,
               7.8685573612e-09f, 0, 3.9062500000e-03f, -128);

    // Op#134: FULLY_CONNECTED
    *(ptensors + 216) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 203), 187, 1, 128,
          (const int8_t*)*(ptensors + 23), (const int32_t*)*(ptensors + 22), *(ptensors + 216),
          0, pscales_q16_op134, -7);

    // Op#135: RESHAPE
    *(ptensors + 217) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 216), *(ptensors + 217), 23936);

    // Op#136: ADD
    *(ptensors + 218) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 217), *(ptensors + 21), *(ptensors + 218), 23936,
           1011, -7, 65425, -5, -4);

    // Op#137: TRANSPOSE
    *(ptensors + 219) = g_activation_pool + g_slot_offsets[2];
    op_transpose_4d(*(ptensors + 218), *(ptensors + 219), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#138: BATCH_MATMUL
    *(ptensors + 220) = g_activation_pool + g_slot_offsets[0];
    op_batch_matmul(*(ptensors + 215), *(ptensors + 219), *(ptensors + 220),
                    1, 8, 187, 187,
                    256, -128, -4, -4);

    // Op#139: TRANSPOSE
    *(ptensors + 221) = g_activation_pool + g_slot_offsets[2];
    op_transpose_4d(*(ptensors + 220), *(ptensors + 221), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#140: RESHAPE
    *(ptensors + 222) = g_activation_pool + g_slot_offsets[0];
    op_copy(*(ptensors + 221), *(ptensors + 222), 23936);

    // Op#141: FULLY_CONNECTED
    *(ptensors + 223) = g_activation_pool + g_slot_offsets[2];
    { static const int32_t pws_q16[1] = {0};
    op_fc(*(ptensors + 222), 187, 128, 1,
          (const int8_t*)*(ptensors + 20), (const int32_t*)*(ptensors + 19), *(ptensors + 223),
          -4, pws_q16, 15);
    }

    // Op#142: ADD
    *(ptensors + 224) = g_activation_pool + g_slot_offsets[0];
    op_add(*(ptensors + 223), *(ptensors + 198), *(ptensors + 224), 187,
           65536, 15, 65536, 15, 15);

    // Op#143: SQUARED_DIFFERENCE
    *(ptensors + 225) = g_activation_pool + g_slot_offsets[2];
    memset(*(ptensors + 225), -128, 187);

    // Op#144: ADD
    *(ptensors + 226) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 225), *(ptensors + 81), *(ptensors + 226), 187,
           65536, -128, 65536, -128, -128);

    // Op#145: SUB
    *(ptensors + 227) = g_activation_pool + g_slot_offsets[2];
    memset(*(ptensors + 227), 0, 187);

    // Op#146: RSQRT
    *(ptensors + 228) = g_activation_pool + g_slot_offsets[1];
    op_rsqrt(*(ptensors + 226), *(ptensors + 228), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#147: MUL
    *(ptensors + 229) = g_activation_pool + g_slot_offsets[3];
    op_mul(*(ptensors + 227), *(ptensors + 228), *(ptensors + 229), 187,
           8127, 0, -128, 0);

    // Op#148: EXPAND_DIMS
    *(ptensors + 230) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 229), *(ptensors + 230), 187);

    // Op#149: CONV_2D
    *(ptensors + 231) = g_activation_pool + g_slot_offsets[3];
    op_fc(*(ptensors + 230), 187, 1, 4,
          (const int8_t*)*(ptensors + 18), (const int32_t*)*(ptensors + 67), *(ptensors + 231),
          0, pscales_q16_op149, 0);

    // Op#150: RESHAPE
    *(ptensors + 232) = g_activation_pool + g_slot_offsets[2];
    op_copy(*(ptensors + 231), *(ptensors + 232), 748);

    // Op#151: EXPAND_DIMS
    *(ptensors + 233) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 232), *(ptensors + 233), 748);

    // Op#152: CONV_2D
    *(ptensors + 234) = g_activation_pool + g_slot_offsets[2];
    { static const int32_t pws_q16[1] = {533};
    op_fc(*(ptensors + 233), 187, 4, 1,
          (const int8_t*)*(ptensors + 17), (const int32_t*)*(ptensors + 62), *(ptensors + 234),
          0, pws_q16, 0);
    }

    // Op#153: RESHAPE
    *(ptensors + 235) = g_activation_pool + g_slot_offsets[3];
    op_copy(*(ptensors + 234), *(ptensors + 235), 187);

    // Op#154: ADD
    *(ptensors + 236) = g_activation_pool + g_slot_offsets[2];
    op_add(*(ptensors + 235), *(ptensors + 16), *(ptensors + 236), 187,
           0, 0, 636, -128, 12);

    // Op#155: ADD
    *(ptensors + 237) = g_activation_pool + g_slot_offsets[3];
    op_add(*(ptensors + 236), *(ptensors + 224), *(ptensors + 237), 187,
           65536, 12, 65536, 15, 12);

    // Op#156: MEAN
    *(ptensors + 238) = g_activation_pool + g_slot_offsets[0];
    op_mean(*(ptensors + 237), *(ptensors + 238),
            187, 1, 1,
            65536, 12, 12);

    // Op#157: FULLY_CONNECTED
    *(ptensors + 239) = g_activation_pool + g_slot_offsets[2];
    op_fc(*(ptensors + 238), 1, 187, 128,
          (const int8_t*)*(ptensors + 15), (const int32_t*)*(ptensors + 14), *(ptensors + 239),
          12, pscales_q16_op157, -128);

    // Op#158: FULLY_CONNECTED
    *(ptensors + 240) = g_activation_pool + g_slot_offsets[0];
    op_fc(*(ptensors + 239), 1, 128, 64,
          (const int8_t*)*(ptensors + 13), (const int32_t*)*(ptensors + 12), *(ptensors + 240),
          -128, pscales_q16_op158, -128);

    // Op#159: FULLY_CONNECTED
    *(ptensors + 241) = g_activation_pool + g_slot_offsets[2];
    op_fc(*(ptensors + 240), 1, 64, 5,
          (const int8_t*)*(ptensors + 11), (const int32_t*)*(ptensors + 10), *(ptensors + 241),
          -128, pscales_q16_op159, 14);

    // Op#160: SOFTMAX
    *(ptensors + 242) = g_activation_pool + g_slot_offsets[0];
    op_softmax(*(ptensors + 241), *(ptensors + 242), 1, 5,
               1.9118081033e-01f, 14, 3.9062500000e-03f, -128);


    // 反量化输出并找预测类别
    int8_t* pout = *(ptensors + 242);
    int t0 = 0;  // pred
    float t1 = -1e9f;  // max_prob
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {
        float t2 = dequantize_int8(*(pout + i), OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        *(poutput_probs + i) = t2;
        if (t2 > t1) {
            t1 = t2;
            t0 = i;
        }
    }
    return t0;
}

// 获取INT8输出 (用于硬件验证)
void ecgformer_get_int8_output(int8_t* poutput) {
    int8_t* psrc = *(ptensors + 242);
    // 使用memcpy替代循环
    memcpy(poutput, psrc, OUTPUT_CLASSES);
}

// ============== 共享库接口 ==============

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

// ============== 主函数 ==============

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {
    init_tensors();
    
    printf("ECGformer INT8 Bare-metal C Implementation\n");
    printf("==========================================\n");
    
    // 测试用随机输入 (使用malloc, 无null检查)
    float* ptest_input = (float*)malloc(INPUT_SIZE << 2);  // INPUT_SIZE * 4
    for (volatile int i = 0; i < INPUT_SIZE; i++) {
        *(ptest_input + i) = ((float)rand() / RAND_MAX - 0.5f);
    }
    
    // 推理
    float* poutput_probs = (float*)malloc(OUTPUT_CLASSES << 2);  // OUTPUT_CLASSES * 4
    int t0 = ecgformer_inference(ptest_input, poutput_probs);
    
    printf("\nPrediction Results:\n");
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {
        printf("  Class %d (%s): %.4f%s\n", i, CLASS_NAMES[i], *(poutput_probs + i),
               i == t0 ? " <-- Predicted" : "");
    }
    
    free(ptest_input);
    free(poutput_probs);
    return 0;
}
#endif
