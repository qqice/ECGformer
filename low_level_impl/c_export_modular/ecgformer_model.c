/**
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
    tensors[1] = (int8_t*)bias_t1;
    tensors[2] = (int8_t*)bias_t2;
    tensors[3] = (int8_t*)bias_t3;
    tensors[4] = (int8_t*)bias_t4;
    tensors[5] = (int8_t*)bias_t5;
    tensors[6] = (int8_t*)bias_t6;
    tensors[7] = (int8_t*)bias_t7;
    tensors[8] = (int8_t*)bias_t8;
    tensors[9] = (int8_t*)bias_t9;
    tensors[10] = (int8_t*)bias_t10;
    tensors[11] = (int8_t*)weight_t11;
    tensors[12] = (int8_t*)bias_t12;
    tensors[13] = (int8_t*)weight_t13;
    tensors[14] = (int8_t*)bias_t14;
    tensors[15] = (int8_t*)weight_t15;
    tensors[16] = (int8_t*)weight_t16;
    tensors[17] = (int8_t*)weight_t17;
    tensors[18] = (int8_t*)weight_t18;
    tensors[19] = (int8_t*)bias_t19;
    tensors[20] = (int8_t*)weight_t20;
    tensors[21] = (int8_t*)weight_t21;
    tensors[22] = (int8_t*)bias_t22;
    tensors[23] = (int8_t*)weight_t23;
    tensors[24] = (int8_t*)weight_t24;
    tensors[25] = (int8_t*)bias_t25;
    tensors[26] = (int8_t*)weight_t26;
    tensors[27] = (int8_t*)weight_t27;
    tensors[28] = (int8_t*)bias_t28;
    tensors[29] = (int8_t*)weight_t29;
    tensors[30] = (int8_t*)weight_t30;
    tensors[31] = (int8_t*)weight_t31;
    tensors[32] = (int8_t*)weight_t32;
    tensors[33] = (int8_t*)bias_t33;
    tensors[34] = (int8_t*)weight_t34;
    tensors[35] = (int8_t*)weight_t35;
    tensors[36] = (int8_t*)bias_t36;
    tensors[37] = (int8_t*)weight_t37;
    tensors[38] = (int8_t*)weight_t38;
    tensors[39] = (int8_t*)bias_t39;
    tensors[40] = (int8_t*)weight_t40;
    tensors[41] = (int8_t*)weight_t41;
    tensors[42] = (int8_t*)bias_t42;
    tensors[43] = (int8_t*)weight_t43;
    tensors[44] = (int8_t*)weight_t44;
    tensors[45] = (int8_t*)weight_t45;
    tensors[46] = (int8_t*)weight_t46;
    tensors[47] = (int8_t*)bias_t47;
    tensors[48] = (int8_t*)weight_t48;
    tensors[49] = (int8_t*)weight_t49;
    tensors[50] = (int8_t*)bias_t50;
    tensors[51] = (int8_t*)weight_t51;
    tensors[52] = (int8_t*)weight_t52;
    tensors[53] = (int8_t*)bias_t53;
    tensors[54] = (int8_t*)weight_t54;
    tensors[55] = (int8_t*)weight_t55;
    tensors[56] = (int8_t*)bias_t56;
    tensors[57] = (int8_t*)weight_t57;
    tensors[58] = (int8_t*)weight_t58;
    tensors[59] = (int8_t*)bias_t59;
    tensors[60] = (int8_t*)bias_t60;
    tensors[61] = (int8_t*)bias_t61;
    tensors[62] = (int8_t*)bias_t62;
    tensors[63] = (int8_t*)weight_t63;
    tensors[64] = (int8_t*)bias_t64;
    tensors[65] = (int8_t*)bias_t65;
    tensors[66] = (int8_t*)bias_t66;
    tensors[67] = (int8_t*)bias_t67;
    tensors[68] = (int8_t*)weight_t68;
    tensors[69] = (int8_t*)bias_t69;
    tensors[70] = (int8_t*)weight_t70;
    tensors[71] = (int8_t*)weight_t71;
    tensors[72] = (int8_t*)bias_t72;
    tensors[73] = (int8_t*)weight_t73;
    tensors[74] = (int8_t*)weight_t74;
    tensors[75] = (int8_t*)weight_t75;
    tensors[76] = (int8_t*)bias_t76;
    tensors[77] = (int8_t*)weight_t77;
    tensors[78] = (int8_t*)weight_t78;
    tensors[79] = (int8_t*)bias_t79;
    tensors[80] = (int8_t*)weight_t80;
    tensors[81] = (int8_t*)weight_t81;
    tensors[0] = &activation_pool[0];
    tensors[82] = &activation_pool[187];
    tensors[83] = &activation_pool[374];
    tensors[84] = &activation_pool[561];
    tensors[85] = &activation_pool[748];
    tensors[86] = &activation_pool[935];
    tensors[87] = &activation_pool[1122];
    tensors[88] = &activation_pool[25058];
    tensors[89] = &activation_pool[48994];
    tensors[90] = &activation_pool[72930];
    tensors[91] = &activation_pool[96866];
    tensors[92] = &activation_pool[120802];
    tensors[93] = &activation_pool[144738];
    tensors[94] = &activation_pool[168674];
    tensors[95] = &activation_pool[192610];
    tensors[96] = &activation_pool[216546];
    tensors[97] = &activation_pool[496298];
    tensors[98] = &activation_pool[776050];
    tensors[99] = &activation_pool[1055802];
    tensors[100] = &activation_pool[1079738];
    tensors[101] = &activation_pool[1103674];
    tensors[102] = &activation_pool[1127610];
    tensors[103] = &activation_pool[1151546];
    tensors[104] = &activation_pool[1175482];
    tensors[105] = &activation_pool[1199418];
    tensors[106] = &activation_pool[1223354];
    tensors[107] = &activation_pool[1223541];
    tensors[108] = &activation_pool[1223728];
    tensors[109] = &activation_pool[1223915];
    tensors[110] = &activation_pool[1224102];
    tensors[111] = &activation_pool[1224289];
    tensors[112] = &activation_pool[1224476];
    tensors[113] = &activation_pool[1224663];
    tensors[114] = &activation_pool[1224850];
    tensors[115] = &activation_pool[1225598];
    tensors[116] = &activation_pool[1226346];
    tensors[117] = &activation_pool[1227094];
    tensors[118] = &activation_pool[1227281];
    tensors[119] = &activation_pool[1227468];
    tensors[120] = &activation_pool[1227655];
    tensors[121] = &activation_pool[1227842];
    tensors[122] = &activation_pool[1228029];
    tensors[123] = &activation_pool[1228216];
    tensors[124] = &activation_pool[1228403];
    tensors[125] = &activation_pool[1228590];
    tensors[126] = &activation_pool[1228777];
    tensors[127] = &activation_pool[1252713];
    tensors[128] = &activation_pool[1276649];
    tensors[129] = &activation_pool[1300585];
    tensors[130] = &activation_pool[1324521];
    tensors[131] = &activation_pool[1348457];
    tensors[132] = &activation_pool[1372393];
    tensors[133] = &activation_pool[1396329];
    tensors[134] = &activation_pool[1420265];
    tensors[135] = &activation_pool[1444201];
    tensors[136] = &activation_pool[1723953];
    tensors[137] = &activation_pool[2003705];
    tensors[138] = &activation_pool[2283457];
    tensors[139] = &activation_pool[2307393];
    tensors[140] = &activation_pool[2331329];
    tensors[141] = &activation_pool[2355265];
    tensors[142] = &activation_pool[2379201];
    tensors[143] = &activation_pool[2403137];
    tensors[144] = &activation_pool[2427073];
    tensors[145] = &activation_pool[2451009];
    tensors[146] = &activation_pool[2451196];
    tensors[147] = &activation_pool[2451383];
    tensors[148] = &activation_pool[2451570];
    tensors[149] = &activation_pool[2451757];
    tensors[150] = &activation_pool[2451944];
    tensors[151] = &activation_pool[2452131];
    tensors[152] = &activation_pool[2452318];
    tensors[153] = &activation_pool[2452505];
    tensors[154] = &activation_pool[2453253];
    tensors[155] = &activation_pool[2454001];
    tensors[156] = &activation_pool[2454749];
    tensors[157] = &activation_pool[2454936];
    tensors[158] = &activation_pool[2455123];
    tensors[159] = &activation_pool[2455310];
    tensors[160] = &activation_pool[2455497];
    tensors[161] = &activation_pool[2455684];
    tensors[162] = &activation_pool[2455871];
    tensors[163] = &activation_pool[2456058];
    tensors[164] = &activation_pool[2456245];
    tensors[165] = &activation_pool[2456432];
    tensors[166] = &activation_pool[2480368];
    tensors[167] = &activation_pool[2504304];
    tensors[168] = &activation_pool[2528240];
    tensors[169] = &activation_pool[2552176];
    tensors[170] = &activation_pool[2576112];
    tensors[171] = &activation_pool[2600048];
    tensors[172] = &activation_pool[2623984];
    tensors[173] = &activation_pool[2647920];
    tensors[174] = &activation_pool[2671856];
    tensors[175] = &activation_pool[2951608];
    tensors[176] = &activation_pool[3231360];
    tensors[177] = &activation_pool[3511112];
    tensors[178] = &activation_pool[3535048];
    tensors[179] = &activation_pool[3558984];
    tensors[180] = &activation_pool[3582920];
    tensors[181] = &activation_pool[3606856];
    tensors[182] = &activation_pool[3630792];
    tensors[183] = &activation_pool[3654728];
    tensors[184] = &activation_pool[3678664];
    tensors[185] = &activation_pool[3678851];
    tensors[186] = &activation_pool[3679038];
    tensors[187] = &activation_pool[3679225];
    tensors[188] = &activation_pool[3679412];
    tensors[189] = &activation_pool[3679599];
    tensors[190] = &activation_pool[3679786];
    tensors[191] = &activation_pool[3679973];
    tensors[192] = &activation_pool[3680160];
    tensors[193] = &activation_pool[3680908];
    tensors[194] = &activation_pool[3681656];
    tensors[195] = &activation_pool[3682404];
    tensors[196] = &activation_pool[3682591];
    tensors[197] = &activation_pool[3682778];
    tensors[198] = &activation_pool[3682965];
    tensors[199] = &activation_pool[3683152];
    tensors[200] = &activation_pool[3683339];
    tensors[201] = &activation_pool[3683526];
    tensors[202] = &activation_pool[3683713];
    tensors[203] = &activation_pool[3683900];
    tensors[204] = &activation_pool[3684087];
    tensors[205] = &activation_pool[3708023];
    tensors[206] = &activation_pool[3731959];
    tensors[207] = &activation_pool[3755895];
    tensors[208] = &activation_pool[3779831];
    tensors[209] = &activation_pool[3803767];
    tensors[210] = &activation_pool[3827703];
    tensors[211] = &activation_pool[3851639];
    tensors[212] = &activation_pool[3875575];
    tensors[213] = &activation_pool[3899511];
    tensors[214] = &activation_pool[4179263];
    tensors[215] = &activation_pool[4459015];
    tensors[216] = &activation_pool[4738767];
    tensors[217] = &activation_pool[4762703];
    tensors[218] = &activation_pool[4786639];
    tensors[219] = &activation_pool[4810575];
    tensors[220] = &activation_pool[4834511];
    tensors[221] = &activation_pool[4858447];
    tensors[222] = &activation_pool[4882383];
    tensors[223] = &activation_pool[4906319];
    tensors[224] = &activation_pool[4906506];
    tensors[225] = &activation_pool[4906693];
    tensors[226] = &activation_pool[4906880];
    tensors[227] = &activation_pool[4907067];
    tensors[228] = &activation_pool[4907254];
    tensors[229] = &activation_pool[4907441];
    tensors[230] = &activation_pool[4907628];
    tensors[231] = &activation_pool[4907815];
    tensors[232] = &activation_pool[4908563];
    tensors[233] = &activation_pool[4909311];
    tensors[234] = &activation_pool[4910059];
    tensors[235] = &activation_pool[4910246];
    tensors[236] = &activation_pool[4910433];
    tensors[237] = &activation_pool[4910620];
    tensors[238] = &activation_pool[4910807];
    tensors[239] = &activation_pool[4910994];
    tensors[240] = &activation_pool[4911122];
    tensors[241] = &activation_pool[4911186];
    tensors[242] = &activation_pool[4911191];
}

// ============== 推理函数 ==============

int ecgformer_inference(const float* input_float, float* output_probs) {
    // 量化输入
    for (int i = 0; i < INPUT_SIZE; i++) {
        tensors[0][i] = quantize_float(input_float[i], INPUT_SCALE, INPUT_ZERO_POINT);
    }
    
    // Op#0: SQUARED_DIFFERENCE
    memset(tensors[82], -128, 187);

    // Op#1: ADD
    op_add(tensors[82], tensors[81], tensors[83], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#2: SUB
    memset(tensors[84], 0, 187);

    // Op#3: RSQRT
    op_rsqrt(tensors[83], tensors[85], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#4: MUL
    op_mul(tensors[84], tensors[85], tensors[86], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#5: FULLY_CONNECTED
    op_fc(tensors[86], 187, 1, 128,
          (const int8_t*)tensors[80], (const int32_t*)tensors[79], tensors[87],
          7.8431368067e-09f, 0, scales_t80, 3.4209706428e-06f, -1);

    // Op#6: RESHAPE
    op_copy(tensors[87], tensors[88], 23936);

    // Op#7: ADD
    op_add(tensors[88], tensors[78], tensors[89], 23936,
           3.4209706428e-06f, -1, 3.9215684033e-09f, 0, 3.4209290334e-06f, -1);

    // Op#8: TRANSPOSE
    op_transpose_4d(tensors[89], tensors[90], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#9: FULLY_CONNECTED
    op_fc(tensors[86], 187, 1, 128,
          (const int8_t*)tensors[77], (const int32_t*)tensors[76], tensors[91],
          7.8431368067e-09f, 0, scales_t77, 3.4717938888e-06f, 0);

    // Op#10: RESHAPE
    op_copy(tensors[91], tensors[92], 23936);

    // Op#11: ADD
    op_add(tensors[92], tensors[75], tensors[93], 23936,
           3.4717938888e-06f, 0, 5.4321230181e-08f, -6, 3.4594715999e-06f, -1);

    // Op#12: MUL
    op_mul(tensors[93], tensors[74], tensors[94], 23936,
           3.4594715999e-06f, -1, 9.8039221484e-04f, -128, 8.6486789996e-07f, -1);

    // Op#13: TRANSPOSE
    op_transpose_4d(tensors[94], tensors[95], 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#14: BATCH_MATMUL
    op_batch_matmul(tensors[90], tensors[95], tensors[96],
                    1, 8, 187, 16,
                    3.4209290334e-06f, -1, 8.6486789996e-07f, -1, 8.5386897553e-09f, -3);

    // Op#15: TRANSPOSE
    op_transpose_4d(tensors[96], tensors[97], 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#16: SOFTMAX
    op_softmax(tensors[97], tensors[98], 1496, 187,
               8.5386897553e-09f, -3, 3.9062500000e-03f, -128);

    // Op#17: FULLY_CONNECTED
    op_fc(tensors[86], 187, 1, 128,
          (const int8_t*)tensors[73], (const int32_t*)tensors[72], tensors[99],
          7.8431368067e-09f, 0, scales_t73, 3.5529558318e-06f, 2);

    // Op#18: RESHAPE
    op_copy(tensors[99], tensors[100], 23936);

    // Op#19: ADD
    op_add(tensors[100], tensors[71], tensors[101], 23936,
           3.5529558318e-06f, 2, 5.5927419453e-05f, -14, 5.7683813793e-05f, -11);

    // Op#20: TRANSPOSE
    op_transpose_4d(tensors[101], tensors[102], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#21: BATCH_MATMUL
    op_batch_matmul(tensors[98], tensors[102], tensors[103],
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 5.7683813793e-05f, -11, 5.7683741034e-05f, -11);

    // Op#22: TRANSPOSE
    op_transpose_4d(tensors[103], tensors[104], 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#23: RESHAPE
    op_copy(tensors[104], tensors[105], 23936);

    // Op#24: FULLY_CONNECTED
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=4.1610049084e-04f;
    op_fc(tensors[105], 187, 128, 1,
          (const int8_t*)tensors[70], (const int32_t*)tensors[69], tensors[106],
          5.7683741034e-05f, -11, ws, 3.5996966064e-02f, 22);
    }

    // Op#25: ADD
    op_add(tensors[106], tensors[0], tensors[107], 187,
           3.5996966064e-02f, 22, 3.5996969789e-02f, 22, 3.5996966064e-02f, 22);

    // Op#26: SQUARED_DIFFERENCE
    memset(tensors[108], -128, 187);

    // Op#27: ADD
    op_add(tensors[108], tensors[81], tensors[109], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#28: SUB
    memset(tensors[110], 0, 187);

    // Op#29: RSQRT
    op_rsqrt(tensors[109], tensors[111], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#30: MUL
    op_mul(tensors[110], tensors[111], tensors[112], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#31: EXPAND_DIMS
    op_copy(tensors[112], tensors[113], 187);

    // Op#32: CONV_2D
    op_fc(tensors[113], 187, 1, 4,
          (const int8_t*)tensors[68], (const int32_t*)tensors[64], tensors[114],
          7.8431368067e-09f, 0, scales_t68, 7.8431368067e-09f, 0);

    // Op#33: RESHAPE
    op_copy(tensors[114], tensors[115], 748);

    // Op#34: EXPAND_DIMS
    op_copy(tensors[115], tensors[116], 748);

    // Op#35: CONV_2D
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=8.5356691852e-03f;
    op_fc(tensors[116], 187, 4, 1,
          (const int8_t*)tensors[63], (const int32_t*)tensors[59], tensors[117],
          7.8431368067e-09f, 0, ws, 7.8431368067e-09f, 0);
    }

    // Op#36: RESHAPE
    op_copy(tensors[117], tensors[118], 187);

    // Op#37: ADD
    op_add(tensors[118], tensors[58], tensors[119], 187,
           7.8431368067e-09f, 0, 3.4954844159e-04f, -128, 3.5996969789e-02f, 20);

    // Op#38: ADD
    op_add(tensors[119], tensors[107], tensors[120], 187,
           3.5996969789e-02f, 20, 3.5996966064e-02f, 22, 3.5996969789e-02f, 20);

    // Op#39: SQUARED_DIFFERENCE
    memset(tensors[121], -128, 187);

    // Op#40: ADD
    op_add(tensors[121], tensors[81], tensors[122], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#41: SUB
    memset(tensors[123], 0, 187);

    // Op#42: RSQRT
    op_rsqrt(tensors[122], tensors[124], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#43: MUL
    op_mul(tensors[123], tensors[124], tensors[125], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#44: FULLY_CONNECTED
    op_fc(tensors[125], 187, 1, 128,
          (const int8_t*)tensors[57], (const int32_t*)tensors[56], tensors[126],
          7.8431368067e-09f, 0, scales_t57, 1.8048147467e-05f, -1);

    // Op#45: RESHAPE
    op_copy(tensors[126], tensors[127], 23936);

    // Op#46: ADD
    op_add(tensors[127], tensors[55], tensors[128], 23936,
           1.8048147467e-05f, -1, 3.9215684033e-09f, 0, 1.8048127458e-05f, -1);

    // Op#47: TRANSPOSE
    op_transpose_4d(tensors[128], tensors[129], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#48: FULLY_CONNECTED
    op_fc(tensors[125], 187, 1, 128,
          (const int8_t*)tensors[54], (const int32_t*)tensors[53], tensors[130],
          7.8431368067e-09f, 0, scales_t54, 1.8206761524e-05f, 1);

    // Op#49: RESHAPE
    op_copy(tensors[130], tensors[131], 23936);

    // Op#50: ADD
    op_add(tensors[131], tensors[52], tensors[132], 23936,
           1.8206761524e-05f, 1, 1.3768246276e-07f, -6, 1.8162952983e-05f, 1);

    // Op#51: MUL
    op_mul(tensors[132], tensors[74], tensors[133], 23936,
           1.8162952983e-05f, 1, 9.8039221484e-04f, -128, 4.5407382459e-06f, 1);

    // Op#52: TRANSPOSE
    op_transpose_4d(tensors[133], tensors[134], 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#53: BATCH_MATMUL
    op_batch_matmul(tensors[129], tensors[134], tensors[135],
                    1, 8, 187, 16,
                    1.8048127458e-05f, -1, 4.5407382459e-06f, 1, 2.1187917199e-08f, 29);

    // Op#54: TRANSPOSE
    op_transpose_4d(tensors[135], tensors[136], 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#55: SOFTMAX
    op_softmax(tensors[136], tensors[137], 1496, 187,
               2.1187917199e-08f, 29, 3.9062500000e-03f, -128);

    // Op#56: FULLY_CONNECTED
    op_fc(tensors[125], 187, 1, 128,
          (const int8_t*)tensors[51], (const int32_t*)tensors[50], tensors[138],
          7.8431368067e-09f, 0, scales_t51, 1.9571607481e-05f, 0);

    // Op#57: RESHAPE
    op_copy(tensors[138], tensors[139], 23936);

    // Op#58: ADD
    op_add(tensors[139], tensors[49], tensors[140], 23936,
           1.9571607481e-05f, 0, 4.9245376431e-05f, -6, 3.7824407627e-05f, 6);

    // Op#59: TRANSPOSE
    op_transpose_4d(tensors[140], tensors[141], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#60: BATCH_MATMUL
    op_batch_matmul(tensors[137], tensors[141], tensors[142],
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 3.7824407627e-05f, 6, 3.7824371248e-05f, 6);

    // Op#61: TRANSPOSE
    op_transpose_4d(tensors[142], tensors[143], 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#62: RESHAPE
    op_copy(tensors[143], tensors[144], 23936);

    // Op#63: FULLY_CONNECTED
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=3.9634574205e-04f;
    op_fc(tensors[144], 187, 128, 1,
          (const int8_t*)tensors[48], (const int32_t*)tensors[47], tensors[145],
          3.7824371248e-05f, 6, ws, 3.5996966064e-02f, 20);
    }

    // Op#64: ADD
    op_add(tensors[145], tensors[120], tensors[146], 187,
           3.5996966064e-02f, 20, 3.5996969789e-02f, 20, 3.5996966064e-02f, 20);

    // Op#65: SQUARED_DIFFERENCE
    memset(tensors[147], -128, 187);

    // Op#66: ADD
    op_add(tensors[147], tensors[81], tensors[148], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#67: SUB
    memset(tensors[149], 0, 187);

    // Op#68: RSQRT
    op_rsqrt(tensors[148], tensors[150], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#69: MUL
    op_mul(tensors[149], tensors[150], tensors[151], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#70: EXPAND_DIMS
    op_copy(tensors[151], tensors[152], 187);

    // Op#71: CONV_2D
    op_fc(tensors[152], 187, 1, 4,
          (const int8_t*)tensors[46], (const int32_t*)tensors[65], tensors[153],
          7.8431368067e-09f, 0, scales_t46, 7.8431368067e-09f, 0);

    // Op#72: RESHAPE
    op_copy(tensors[153], tensors[154], 748);

    // Op#73: EXPAND_DIMS
    op_copy(tensors[154], tensors[155], 748);

    // Op#74: CONV_2D
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=4.4033546001e-03f;
    op_fc(tensors[155], 187, 4, 1,
          (const int8_t*)tensors[45], (const int32_t*)tensors[60], tensors[156],
          7.8431368067e-09f, 0, ws, 7.8431368067e-09f, 0);
    }

    // Op#75: RESHAPE
    op_copy(tensors[156], tensors[157], 187);

    // Op#76: ADD
    op_add(tensors[157], tensors[44], tensors[158], 187,
           7.8431368067e-09f, 0, 3.4954820876e-04f, -128, 3.5996966064e-02f, 17);

    // Op#77: ADD
    op_add(tensors[158], tensors[146], tensors[159], 187,
           3.5996966064e-02f, 17, 3.5996966064e-02f, 20, 3.5996966064e-02f, 17);

    // Op#78: SQUARED_DIFFERENCE
    memset(tensors[160], -128, 187);

    // Op#79: ADD
    op_add(tensors[160], tensors[81], tensors[161], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#80: SUB
    memset(tensors[162], 0, 187);

    // Op#81: RSQRT
    op_rsqrt(tensors[161], tensors[163], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#82: MUL
    op_mul(tensors[162], tensors[163], tensors[164], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#83: FULLY_CONNECTED
    op_fc(tensors[164], 187, 1, 128,
          (const int8_t*)tensors[43], (const int32_t*)tensors[42], tensors[165],
          7.8431368067e-09f, 0, scales_t43, 8.5654037321e-06f, -1);

    // Op#84: RESHAPE
    op_copy(tensors[165], tensors[166], 23936);

    // Op#85: ADD
    op_add(tensors[166], tensors[41], tensors[167], 23936,
           8.5654037321e-06f, -1, 3.9215684033e-09f, 0, 8.5653437054e-06f, -1);

    // Op#86: TRANSPOSE
    op_transpose_4d(tensors[167], tensors[168], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#87: FULLY_CONNECTED
    op_fc(tensors[164], 187, 1, 128,
          (const int8_t*)tensors[40], (const int32_t*)tensors[39], tensors[169],
          7.8431368067e-09f, 0, scales_t40, 8.8167744252e-06f, -2);

    // Op#88: RESHAPE
    op_copy(tensors[169], tensors[170], 23936);

    // Op#89: ADD
    op_add(tensors[170], tensors[38], tensors[171], 23936,
           8.8167744252e-06f, -2, 8.5421326901e-08f, -1, 8.8170627350e-06f, -2);

    // Op#90: MUL
    op_mul(tensors[171], tensors[74], tensors[172], 23936,
           8.8170627350e-06f, -2, 9.8039221484e-04f, -128, 2.2042656838e-06f, -2);

    // Op#91: TRANSPOSE
    op_transpose_4d(tensors[172], tensors[173], 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#92: BATCH_MATMUL
    op_batch_matmul(tensors[168], tensors[173], tensors[174],
                    1, 8, 187, 16,
                    8.5653437054e-06f, -1, 2.2042656838e-06f, -2, 4.2088466046e-09f, -73);

    // Op#93: TRANSPOSE
    op_transpose_4d(tensors[174], tensors[175], 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#94: SOFTMAX
    op_softmax(tensors[175], tensors[176], 1496, 187,
               4.2088466046e-09f, -73, 3.9062500000e-03f, -128);

    // Op#95: FULLY_CONNECTED
    op_fc(tensors[164], 187, 1, 128,
          (const int8_t*)tensors[37], (const int32_t*)tensors[36], tensors[177],
          7.8431368067e-09f, 0, scales_t37, 8.8702890935e-06f, -2);

    // Op#96: RESHAPE
    op_copy(tensors[177], tensors[178], 23936);

    // Op#97: ADD
    op_add(tensors[178], tensors[35], tensors[179], 23936,
           8.8702890935e-06f, -2, 4.1749022785e-05f, -8, 4.3648789870e-05f, 0);

    // Op#98: TRANSPOSE
    op_transpose_4d(tensors[179], tensors[180], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#99: BATCH_MATMUL
    op_batch_matmul(tensors[176], tensors[180], tensors[181],
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 4.3648789870e-05f, 0, 4.3648738938e-05f, 0);

    // Op#100: TRANSPOSE
    op_transpose_4d(tensors[181], tensors[182], 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#101: RESHAPE
    op_copy(tensors[182], tensors[183], 23936);

    // Op#102: FULLY_CONNECTED
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=2.1708158602e-04f;
    op_fc(tensors[183], 187, 128, 1,
          (const int8_t*)tensors[34], (const int32_t*)tensors[33], tensors[184],
          4.3648738938e-05f, 0, ws, 3.5996966064e-02f, 17);
    }

    // Op#103: ADD
    op_add(tensors[184], tensors[159], tensors[185], 187,
           3.5996966064e-02f, 17, 3.5996966064e-02f, 17, 3.5996966064e-02f, 17);

    // Op#104: SQUARED_DIFFERENCE
    memset(tensors[186], -128, 187);

    // Op#105: ADD
    op_add(tensors[186], tensors[81], tensors[187], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#106: SUB
    memset(tensors[188], 0, 187);

    // Op#107: RSQRT
    op_rsqrt(tensors[187], tensors[189], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#108: MUL
    op_mul(tensors[188], tensors[189], tensors[190], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#109: EXPAND_DIMS
    op_copy(tensors[190], tensors[191], 187);

    // Op#110: CONV_2D
    op_fc(tensors[191], 187, 1, 4,
          (const int8_t*)tensors[32], (const int32_t*)tensors[66], tensors[192],
          7.8431368067e-09f, 0, scales_t32, 7.8431368067e-09f, 0);

    // Op#111: RESHAPE
    op_copy(tensors[192], tensors[193], 748);

    // Op#112: EXPAND_DIMS
    op_copy(tensors[193], tensors[194], 748);

    // Op#113: CONV_2D
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=3.3950232901e-03f;
    op_fc(tensors[194], 187, 4, 1,
          (const int8_t*)tensors[31], (const int32_t*)tensors[61], tensors[195],
          7.8431368067e-09f, 0, ws, 7.8431368067e-09f, 0);
    }

    // Op#114: RESHAPE
    op_copy(tensors[195], tensors[196], 187);

    // Op#115: ADD
    op_add(tensors[196], tensors[30], tensors[197], 187,
           7.8431368067e-09f, 0, 3.4954832518e-04f, -128, 3.5996966064e-02f, 15);

    // Op#116: ADD
    op_add(tensors[197], tensors[185], tensors[198], 187,
           3.5996966064e-02f, 15, 3.5996966064e-02f, 17, 3.5996966064e-02f, 15);

    // Op#117: SQUARED_DIFFERENCE
    memset(tensors[199], -128, 187);

    // Op#118: ADD
    op_add(tensors[199], tensors[81], tensors[200], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#119: SUB
    memset(tensors[201], 0, 187);

    // Op#120: RSQRT
    op_rsqrt(tensors[200], tensors[202], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#121: MUL
    op_mul(tensors[201], tensors[202], tensors[203], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#122: FULLY_CONNECTED
    op_fc(tensors[203], 187, 1, 128,
          (const int8_t*)tensors[29], (const int32_t*)tensors[28], tensors[204],
          7.8431368067e-09f, 0, scales_t29, 4.5233937840e-07f, -2);

    // Op#123: RESHAPE
    op_copy(tensors[204], tensors[205], 23936);

    // Op#124: ADD
    op_add(tensors[205], tensors[27], tensors[206], 23936,
           4.5233937840e-07f, -2, 3.9215684033e-09f, 0, 4.5233127821e-07f, -2);

    // Op#125: TRANSPOSE
    op_transpose_4d(tensors[206], tensors[207], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#126: FULLY_CONNECTED
    op_fc(tensors[203], 187, 1, 128,
          (const int8_t*)tensors[26], (const int32_t*)tensors[25], tensors[208],
          7.8431368067e-09f, 0, scales_t26, 4.5768948098e-07f, -1);

    // Op#127: RESHAPE
    op_copy(tensors[208], tensors[209], 23936);

    // Op#128: ADD
    op_add(tensors[209], tensors[24], tensors[210], 23936,
           4.5768948098e-07f, -1, 1.2812115813e-07f, 5, 4.8657540219e-07f, 1);

    // Op#129: MUL
    op_mul(tensors[210], tensors[74], tensors[211], 23936,
           4.8657540219e-07f, 1, 9.8039221484e-04f, -128, 1.2164385055e-07f, 1);

    // Op#130: TRANSPOSE
    op_transpose_4d(tensors[211], tensors[212], 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#131: BATCH_MATMUL
    op_batch_matmul(tensors[207], tensors[212], tensors[213],
                    1, 8, 187, 16,
                    4.5233127821e-07f, -2, 1.2164385055e-07f, 1, 7.8685573612e-09f, 0);

    // Op#132: TRANSPOSE
    op_transpose_4d(tensors[213], tensors[214], 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#133: SOFTMAX
    op_softmax(tensors[214], tensors[215], 1496, 187,
               7.8685573612e-09f, 0, 3.9062500000e-03f, -128);

    // Op#134: FULLY_CONNECTED
    op_fc(tensors[203], 187, 1, 128,
          (const int8_t*)tensors[23], (const int32_t*)tensors[22], tensors[216],
          7.8431368067e-09f, 0, scales_t23, 4.5547096761e-07f, -7);

    // Op#135: RESHAPE
    op_copy(tensors[216], tensors[217], 23936);

    // Op#136: ADD
    op_add(tensors[217], tensors[21], tensors[218], 23936,
           4.5547096761e-07f, -7, 2.9470964364e-05f, -5, 2.9520904718e-05f, -4);

    // Op#137: TRANSPOSE
    op_transpose_4d(tensors[218], tensors[219], 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#138: BATCH_MATMUL
    op_batch_matmul(tensors[215], tensors[219], tensors[220],
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 2.9520904718e-05f, -4, 2.9520893804e-05f, -4);

    // Op#139: TRANSPOSE
    op_transpose_4d(tensors[220], tensors[221], 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#140: RESHAPE
    op_copy(tensors[221], tensors[222], 23936);

    // Op#141: FULLY_CONNECTED
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=2.9882427771e-04f;
    op_fc(tensors[222], 187, 128, 1,
          (const int8_t*)tensors[20], (const int32_t*)tensors[19], tensors[223],
          2.9520893804e-05f, -4, ws, 3.5996966064e-02f, 15);
    }

    // Op#142: ADD
    op_add(tensors[223], tensors[198], tensors[224], 187,
           3.5996966064e-02f, 15, 3.5996966064e-02f, 15, 3.5996966064e-02f, 15);

    // Op#143: SQUARED_DIFFERENCE
    memset(tensors[225], -128, 187);

    // Op#144: ADD
    op_add(tensors[225], tensors[81], tensors[226], 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#145: SUB
    memset(tensors[227], 0, 187);

    // Op#146: RSQRT
    op_rsqrt(tensors[226], tensors[228], 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#147: MUL
    op_mul(tensors[227], tensors[228], tensors[229], 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#148: EXPAND_DIMS
    op_copy(tensors[229], tensors[230], 187);

    // Op#149: CONV_2D
    op_fc(tensors[230], 187, 1, 4,
          (const int8_t*)tensors[18], (const int32_t*)tensors[67], tensors[231],
          7.8431368067e-09f, 0, scales_t18, 7.8431368067e-09f, 0);

    // Op#150: RESHAPE
    op_copy(tensors[231], tensors[232], 748);

    // Op#151: EXPAND_DIMS
    op_copy(tensors[232], tensors[233], 748);

    // Op#152: CONV_2D
    { float ws[1]; for(int i=0;i<1;i++) ws[i]=8.1268763170e-03f;
    op_fc(tensors[233], 187, 4, 1,
          (const int8_t*)tensors[17], (const int32_t*)tensors[62], tensors[234],
          7.8431368067e-09f, 0, ws, 7.8431368067e-09f, 0);
    }

    // Op#153: RESHAPE
    op_copy(tensors[234], tensors[235], 187);

    // Op#154: ADD
    op_add(tensors[235], tensors[16], tensors[236], 187,
           7.8431368067e-09f, 0, 3.4954876173e-04f, -128, 3.5996966064e-02f, 12);

    // Op#155: ADD
    op_add(tensors[236], tensors[224], tensors[237], 187,
           3.5996966064e-02f, 12, 3.5996966064e-02f, 15, 3.5996966064e-02f, 12);

    // Op#156: MEAN
    op_mean(tensors[237], tensors[238],
            187, 1, 1,
            3.5996966064e-02f, 12, 3.5996966064e-02f, 12);

    // Op#157: FULLY_CONNECTED
    op_fc(tensors[238], 1, 187, 128,
          (const int8_t*)tensors[15], (const int32_t*)tensors[14], tensors[239],
          3.5996966064e-02f, 12, scales_t15, 2.6135431603e-02f, -128);

    // Op#158: FULLY_CONNECTED
    op_fc(tensors[239], 1, 128, 64,
          (const int8_t*)tensors[13], (const int32_t*)tensors[12], tensors[240],
          2.6135431603e-02f, -128, scales_t13, 3.1560700387e-02f, -128);

    // Op#159: FULLY_CONNECTED
    op_fc(tensors[240], 1, 64, 5,
          (const int8_t*)tensors[11], (const int32_t*)tensors[10], tensors[241],
          3.1560700387e-02f, -128, scales_t11, 1.9118081033e-01f, 14);

    // Op#160: SOFTMAX
    op_softmax(tensors[241], tensors[242], 1, 5,
               1.9118081033e-01f, 14, 3.9062500000e-03f, -128);


    // 反量化输出并找预测类别
    int pred = 0;
    float max_prob = -1e9f;
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        output_probs[i] = dequantize_int8(tensors[242][i], OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        if (output_probs[i] > max_prob) {
            max_prob = output_probs[i];
            pred = i;
        }
    }
    return pred;
}

// 获取INT8输出（用于验证）
void ecgformer_get_int8_output(int8_t* output) {
    memcpy(output, tensors[242], OUTPUT_CLASSES);
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
    
    printf("ECGformer INT8 模块化C实现\n");
    printf("==============================\n");
    
    // 测试用随机输入
    float test_input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        test_input[i] = ((float)rand() / RAND_MAX - 0.5f);
    }
    
    // 推理
    float output_probs[OUTPUT_CLASSES];
    int pred = ecgformer_inference(test_input, output_probs);
    
    printf("\n预测结果:\n");
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        printf("  类别 %d (%s): %.4f%s\n", i, CLASS_NAMES[i], output_probs[i],
               i == pred ? " <-- 预测" : "");
    }
    
    return 0;
}
#endif
