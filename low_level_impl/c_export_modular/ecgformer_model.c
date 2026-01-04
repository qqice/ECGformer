/**
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
    *(ptensors + 0) = g_activation_pool + 0;
    *(ptensors + 82) = g_activation_pool + 187;
    *(ptensors + 83) = g_activation_pool + 374;
    *(ptensors + 84) = g_activation_pool + 561;
    *(ptensors + 85) = g_activation_pool + 748;
    *(ptensors + 86) = g_activation_pool + 935;
    *(ptensors + 87) = g_activation_pool + 1122;
    *(ptensors + 88) = g_activation_pool + 25058;
    *(ptensors + 89) = g_activation_pool + 48994;
    *(ptensors + 90) = g_activation_pool + 72930;
    *(ptensors + 91) = g_activation_pool + 96866;
    *(ptensors + 92) = g_activation_pool + 120802;
    *(ptensors + 93) = g_activation_pool + 144738;
    *(ptensors + 94) = g_activation_pool + 168674;
    *(ptensors + 95) = g_activation_pool + 192610;
    *(ptensors + 96) = g_activation_pool + 216546;
    *(ptensors + 97) = g_activation_pool + 496298;
    *(ptensors + 98) = g_activation_pool + 776050;
    *(ptensors + 99) = g_activation_pool + 1055802;
    *(ptensors + 100) = g_activation_pool + 1079738;
    *(ptensors + 101) = g_activation_pool + 1103674;
    *(ptensors + 102) = g_activation_pool + 1127610;
    *(ptensors + 103) = g_activation_pool + 1151546;
    *(ptensors + 104) = g_activation_pool + 1175482;
    *(ptensors + 105) = g_activation_pool + 1199418;
    *(ptensors + 106) = g_activation_pool + 1223354;
    *(ptensors + 107) = g_activation_pool + 1223541;
    *(ptensors + 108) = g_activation_pool + 1223728;
    *(ptensors + 109) = g_activation_pool + 1223915;
    *(ptensors + 110) = g_activation_pool + 1224102;
    *(ptensors + 111) = g_activation_pool + 1224289;
    *(ptensors + 112) = g_activation_pool + 1224476;
    *(ptensors + 113) = g_activation_pool + 1224663;
    *(ptensors + 114) = g_activation_pool + 1224850;
    *(ptensors + 115) = g_activation_pool + 1225598;
    *(ptensors + 116) = g_activation_pool + 1226346;
    *(ptensors + 117) = g_activation_pool + 1227094;
    *(ptensors + 118) = g_activation_pool + 1227281;
    *(ptensors + 119) = g_activation_pool + 1227468;
    *(ptensors + 120) = g_activation_pool + 1227655;
    *(ptensors + 121) = g_activation_pool + 1227842;
    *(ptensors + 122) = g_activation_pool + 1228029;
    *(ptensors + 123) = g_activation_pool + 1228216;
    *(ptensors + 124) = g_activation_pool + 1228403;
    *(ptensors + 125) = g_activation_pool + 1228590;
    *(ptensors + 126) = g_activation_pool + 1228777;
    *(ptensors + 127) = g_activation_pool + 1252713;
    *(ptensors + 128) = g_activation_pool + 1276649;
    *(ptensors + 129) = g_activation_pool + 1300585;
    *(ptensors + 130) = g_activation_pool + 1324521;
    *(ptensors + 131) = g_activation_pool + 1348457;
    *(ptensors + 132) = g_activation_pool + 1372393;
    *(ptensors + 133) = g_activation_pool + 1396329;
    *(ptensors + 134) = g_activation_pool + 1420265;
    *(ptensors + 135) = g_activation_pool + 1444201;
    *(ptensors + 136) = g_activation_pool + 1723953;
    *(ptensors + 137) = g_activation_pool + 2003705;
    *(ptensors + 138) = g_activation_pool + 2283457;
    *(ptensors + 139) = g_activation_pool + 2307393;
    *(ptensors + 140) = g_activation_pool + 2331329;
    *(ptensors + 141) = g_activation_pool + 2355265;
    *(ptensors + 142) = g_activation_pool + 2379201;
    *(ptensors + 143) = g_activation_pool + 2403137;
    *(ptensors + 144) = g_activation_pool + 2427073;
    *(ptensors + 145) = g_activation_pool + 2451009;
    *(ptensors + 146) = g_activation_pool + 2451196;
    *(ptensors + 147) = g_activation_pool + 2451383;
    *(ptensors + 148) = g_activation_pool + 2451570;
    *(ptensors + 149) = g_activation_pool + 2451757;
    *(ptensors + 150) = g_activation_pool + 2451944;
    *(ptensors + 151) = g_activation_pool + 2452131;
    *(ptensors + 152) = g_activation_pool + 2452318;
    *(ptensors + 153) = g_activation_pool + 2452505;
    *(ptensors + 154) = g_activation_pool + 2453253;
    *(ptensors + 155) = g_activation_pool + 2454001;
    *(ptensors + 156) = g_activation_pool + 2454749;
    *(ptensors + 157) = g_activation_pool + 2454936;
    *(ptensors + 158) = g_activation_pool + 2455123;
    *(ptensors + 159) = g_activation_pool + 2455310;
    *(ptensors + 160) = g_activation_pool + 2455497;
    *(ptensors + 161) = g_activation_pool + 2455684;
    *(ptensors + 162) = g_activation_pool + 2455871;
    *(ptensors + 163) = g_activation_pool + 2456058;
    *(ptensors + 164) = g_activation_pool + 2456245;
    *(ptensors + 165) = g_activation_pool + 2456432;
    *(ptensors + 166) = g_activation_pool + 2480368;
    *(ptensors + 167) = g_activation_pool + 2504304;
    *(ptensors + 168) = g_activation_pool + 2528240;
    *(ptensors + 169) = g_activation_pool + 2552176;
    *(ptensors + 170) = g_activation_pool + 2576112;
    *(ptensors + 171) = g_activation_pool + 2600048;
    *(ptensors + 172) = g_activation_pool + 2623984;
    *(ptensors + 173) = g_activation_pool + 2647920;
    *(ptensors + 174) = g_activation_pool + 2671856;
    *(ptensors + 175) = g_activation_pool + 2951608;
    *(ptensors + 176) = g_activation_pool + 3231360;
    *(ptensors + 177) = g_activation_pool + 3511112;
    *(ptensors + 178) = g_activation_pool + 3535048;
    *(ptensors + 179) = g_activation_pool + 3558984;
    *(ptensors + 180) = g_activation_pool + 3582920;
    *(ptensors + 181) = g_activation_pool + 3606856;
    *(ptensors + 182) = g_activation_pool + 3630792;
    *(ptensors + 183) = g_activation_pool + 3654728;
    *(ptensors + 184) = g_activation_pool + 3678664;
    *(ptensors + 185) = g_activation_pool + 3678851;
    *(ptensors + 186) = g_activation_pool + 3679038;
    *(ptensors + 187) = g_activation_pool + 3679225;
    *(ptensors + 188) = g_activation_pool + 3679412;
    *(ptensors + 189) = g_activation_pool + 3679599;
    *(ptensors + 190) = g_activation_pool + 3679786;
    *(ptensors + 191) = g_activation_pool + 3679973;
    *(ptensors + 192) = g_activation_pool + 3680160;
    *(ptensors + 193) = g_activation_pool + 3680908;
    *(ptensors + 194) = g_activation_pool + 3681656;
    *(ptensors + 195) = g_activation_pool + 3682404;
    *(ptensors + 196) = g_activation_pool + 3682591;
    *(ptensors + 197) = g_activation_pool + 3682778;
    *(ptensors + 198) = g_activation_pool + 3682965;
    *(ptensors + 199) = g_activation_pool + 3683152;
    *(ptensors + 200) = g_activation_pool + 3683339;
    *(ptensors + 201) = g_activation_pool + 3683526;
    *(ptensors + 202) = g_activation_pool + 3683713;
    *(ptensors + 203) = g_activation_pool + 3683900;
    *(ptensors + 204) = g_activation_pool + 3684087;
    *(ptensors + 205) = g_activation_pool + 3708023;
    *(ptensors + 206) = g_activation_pool + 3731959;
    *(ptensors + 207) = g_activation_pool + 3755895;
    *(ptensors + 208) = g_activation_pool + 3779831;
    *(ptensors + 209) = g_activation_pool + 3803767;
    *(ptensors + 210) = g_activation_pool + 3827703;
    *(ptensors + 211) = g_activation_pool + 3851639;
    *(ptensors + 212) = g_activation_pool + 3875575;
    *(ptensors + 213) = g_activation_pool + 3899511;
    *(ptensors + 214) = g_activation_pool + 4179263;
    *(ptensors + 215) = g_activation_pool + 4459015;
    *(ptensors + 216) = g_activation_pool + 4738767;
    *(ptensors + 217) = g_activation_pool + 4762703;
    *(ptensors + 218) = g_activation_pool + 4786639;
    *(ptensors + 219) = g_activation_pool + 4810575;
    *(ptensors + 220) = g_activation_pool + 4834511;
    *(ptensors + 221) = g_activation_pool + 4858447;
    *(ptensors + 222) = g_activation_pool + 4882383;
    *(ptensors + 223) = g_activation_pool + 4906319;
    *(ptensors + 224) = g_activation_pool + 4906506;
    *(ptensors + 225) = g_activation_pool + 4906693;
    *(ptensors + 226) = g_activation_pool + 4906880;
    *(ptensors + 227) = g_activation_pool + 4907067;
    *(ptensors + 228) = g_activation_pool + 4907254;
    *(ptensors + 229) = g_activation_pool + 4907441;
    *(ptensors + 230) = g_activation_pool + 4907628;
    *(ptensors + 231) = g_activation_pool + 4907815;
    *(ptensors + 232) = g_activation_pool + 4908563;
    *(ptensors + 233) = g_activation_pool + 4909311;
    *(ptensors + 234) = g_activation_pool + 4910059;
    *(ptensors + 235) = g_activation_pool + 4910246;
    *(ptensors + 236) = g_activation_pool + 4910433;
    *(ptensors + 237) = g_activation_pool + 4910620;
    *(ptensors + 238) = g_activation_pool + 4910807;
    *(ptensors + 239) = g_activation_pool + 4910994;
    *(ptensors + 240) = g_activation_pool + 4911122;
    *(ptensors + 241) = g_activation_pool + 4911186;
    *(ptensors + 242) = g_activation_pool + 4911191;
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
    memset(*(ptensors + 82), -128, 187);

    // Op#1: ADD
    op_add(*(ptensors + 82), *(ptensors + 81), *(ptensors + 83), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#2: SUB
    memset(*(ptensors + 84), 0, 187);

    // Op#3: RSQRT
    op_rsqrt(*(ptensors + 83), *(ptensors + 85), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#4: MUL
    op_mul(*(ptensors + 84), *(ptensors + 85), *(ptensors + 86), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#5: FULLY_CONNECTED
    op_fc(*(ptensors + 86), 187, 1, 128,
          (const int8_t*)*(ptensors + 80), (const int32_t*)*(ptensors + 79), *(ptensors + 87),
          7.8431368067e-09f, 0, pscales_t80, 3.4209706428e-06f, -1);

    // Op#6: RESHAPE
    op_copy(*(ptensors + 87), *(ptensors + 88), 23936);

    // Op#7: ADD
    op_add(*(ptensors + 88), *(ptensors + 78), *(ptensors + 89), 23936,
           3.4209706428e-06f, -1, 3.9215684033e-09f, 0, 3.4209290334e-06f, -1);

    // Op#8: TRANSPOSE
    op_transpose_4d(*(ptensors + 89), *(ptensors + 90), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#9: FULLY_CONNECTED
    op_fc(*(ptensors + 86), 187, 1, 128,
          (const int8_t*)*(ptensors + 77), (const int32_t*)*(ptensors + 76), *(ptensors + 91),
          7.8431368067e-09f, 0, pscales_t77, 3.4717938888e-06f, 0);

    // Op#10: RESHAPE
    op_copy(*(ptensors + 91), *(ptensors + 92), 23936);

    // Op#11: ADD
    op_add(*(ptensors + 92), *(ptensors + 75), *(ptensors + 93), 23936,
           3.4717938888e-06f, 0, 5.4321230181e-08f, -6, 3.4594715999e-06f, -1);

    // Op#12: MUL
    op_mul(*(ptensors + 93), *(ptensors + 74), *(ptensors + 94), 23936,
           3.4594715999e-06f, -1, 9.8039221484e-04f, -128, 8.6486789996e-07f, -1);

    // Op#13: TRANSPOSE
    op_transpose_4d(*(ptensors + 94), *(ptensors + 95), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#14: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 90), *(ptensors + 95), *(ptensors + 96),
                    1, 8, 187, 16,
                    3.4209290334e-06f, -1, 8.6486789996e-07f, -1, 8.5386897553e-09f, -3);

    // Op#15: TRANSPOSE
    op_transpose_4d(*(ptensors + 96), *(ptensors + 97), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#16: SOFTMAX
    op_softmax(*(ptensors + 97), *(ptensors + 98), 1496, 187,
               8.5386897553e-09f, -3, 3.9062500000e-03f, -128);

    // Op#17: FULLY_CONNECTED
    op_fc(*(ptensors + 86), 187, 1, 128,
          (const int8_t*)*(ptensors + 73), (const int32_t*)*(ptensors + 72), *(ptensors + 99),
          7.8431368067e-09f, 0, pscales_t73, 3.5529558318e-06f, 2);

    // Op#18: RESHAPE
    op_copy(*(ptensors + 99), *(ptensors + 100), 23936);

    // Op#19: ADD
    op_add(*(ptensors + 100), *(ptensors + 71), *(ptensors + 101), 23936,
           3.5529558318e-06f, 2, 5.5927419453e-05f, -14, 5.7683813793e-05f, -11);

    // Op#20: TRANSPOSE
    op_transpose_4d(*(ptensors + 101), *(ptensors + 102), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#21: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 98), *(ptensors + 102), *(ptensors + 103),
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 5.7683813793e-05f, -11, 5.7683741034e-05f, -11);

    // Op#22: TRANSPOSE
    op_transpose_4d(*(ptensors + 103), *(ptensors + 104), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#23: RESHAPE
    op_copy(*(ptensors + 104), *(ptensors + 105), 23936);

    // Op#24: FULLY_CONNECTED
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=4.1610049084e-04f;
    op_fc(*(ptensors + 105), 187, 128, 1,
          (const int8_t*)*(ptensors + 70), (const int32_t*)*(ptensors + 69), *(ptensors + 106),
          5.7683741034e-05f, -11, pws, 3.5996966064e-02f, 22);
      free(pws); }

    // Op#25: ADD
    op_add(*(ptensors + 106), *(ptensors + 0), *(ptensors + 107), 187,
           3.5996966064e-02f, 22, 3.5996969789e-02f, 22, 3.5996966064e-02f, 22);

    // Op#26: SQUARED_DIFFERENCE
    memset(*(ptensors + 108), -128, 187);

    // Op#27: ADD
    op_add(*(ptensors + 108), *(ptensors + 81), *(ptensors + 109), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#28: SUB
    memset(*(ptensors + 110), 0, 187);

    // Op#29: RSQRT
    op_rsqrt(*(ptensors + 109), *(ptensors + 111), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#30: MUL
    op_mul(*(ptensors + 110), *(ptensors + 111), *(ptensors + 112), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#31: EXPAND_DIMS
    op_copy(*(ptensors + 112), *(ptensors + 113), 187);

    // Op#32: CONV_2D
    op_fc(*(ptensors + 113), 187, 1, 4,
          (const int8_t*)*(ptensors + 68), (const int32_t*)*(ptensors + 64), *(ptensors + 114),
          7.8431368067e-09f, 0, pscales_t68, 7.8431368067e-09f, 0);

    // Op#33: RESHAPE
    op_copy(*(ptensors + 114), *(ptensors + 115), 748);

    // Op#34: EXPAND_DIMS
    op_copy(*(ptensors + 115), *(ptensors + 116), 748);

    // Op#35: CONV_2D
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=8.5356691852e-03f;
    op_fc(*(ptensors + 116), 187, 4, 1,
          (const int8_t*)*(ptensors + 63), (const int32_t*)*(ptensors + 59), *(ptensors + 117),
          7.8431368067e-09f, 0, pws, 7.8431368067e-09f, 0);
      free(pws); }

    // Op#36: RESHAPE
    op_copy(*(ptensors + 117), *(ptensors + 118), 187);

    // Op#37: ADD
    op_add(*(ptensors + 118), *(ptensors + 58), *(ptensors + 119), 187,
           7.8431368067e-09f, 0, 3.4954844159e-04f, -128, 3.5996969789e-02f, 20);

    // Op#38: ADD
    op_add(*(ptensors + 119), *(ptensors + 107), *(ptensors + 120), 187,
           3.5996969789e-02f, 20, 3.5996966064e-02f, 22, 3.5996969789e-02f, 20);

    // Op#39: SQUARED_DIFFERENCE
    memset(*(ptensors + 121), -128, 187);

    // Op#40: ADD
    op_add(*(ptensors + 121), *(ptensors + 81), *(ptensors + 122), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#41: SUB
    memset(*(ptensors + 123), 0, 187);

    // Op#42: RSQRT
    op_rsqrt(*(ptensors + 122), *(ptensors + 124), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#43: MUL
    op_mul(*(ptensors + 123), *(ptensors + 124), *(ptensors + 125), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#44: FULLY_CONNECTED
    op_fc(*(ptensors + 125), 187, 1, 128,
          (const int8_t*)*(ptensors + 57), (const int32_t*)*(ptensors + 56), *(ptensors + 126),
          7.8431368067e-09f, 0, pscales_t57, 1.8048147467e-05f, -1);

    // Op#45: RESHAPE
    op_copy(*(ptensors + 126), *(ptensors + 127), 23936);

    // Op#46: ADD
    op_add(*(ptensors + 127), *(ptensors + 55), *(ptensors + 128), 23936,
           1.8048147467e-05f, -1, 3.9215684033e-09f, 0, 1.8048127458e-05f, -1);

    // Op#47: TRANSPOSE
    op_transpose_4d(*(ptensors + 128), *(ptensors + 129), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#48: FULLY_CONNECTED
    op_fc(*(ptensors + 125), 187, 1, 128,
          (const int8_t*)*(ptensors + 54), (const int32_t*)*(ptensors + 53), *(ptensors + 130),
          7.8431368067e-09f, 0, pscales_t54, 1.8206761524e-05f, 1);

    // Op#49: RESHAPE
    op_copy(*(ptensors + 130), *(ptensors + 131), 23936);

    // Op#50: ADD
    op_add(*(ptensors + 131), *(ptensors + 52), *(ptensors + 132), 23936,
           1.8206761524e-05f, 1, 1.3768246276e-07f, -6, 1.8162952983e-05f, 1);

    // Op#51: MUL
    op_mul(*(ptensors + 132), *(ptensors + 74), *(ptensors + 133), 23936,
           1.8162952983e-05f, 1, 9.8039221484e-04f, -128, 4.5407382459e-06f, 1);

    // Op#52: TRANSPOSE
    op_transpose_4d(*(ptensors + 133), *(ptensors + 134), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#53: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 129), *(ptensors + 134), *(ptensors + 135),
                    1, 8, 187, 16,
                    1.8048127458e-05f, -1, 4.5407382459e-06f, 1, 2.1187917199e-08f, 29);

    // Op#54: TRANSPOSE
    op_transpose_4d(*(ptensors + 135), *(ptensors + 136), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#55: SOFTMAX
    op_softmax(*(ptensors + 136), *(ptensors + 137), 1496, 187,
               2.1187917199e-08f, 29, 3.9062500000e-03f, -128);

    // Op#56: FULLY_CONNECTED
    op_fc(*(ptensors + 125), 187, 1, 128,
          (const int8_t*)*(ptensors + 51), (const int32_t*)*(ptensors + 50), *(ptensors + 138),
          7.8431368067e-09f, 0, pscales_t51, 1.9571607481e-05f, 0);

    // Op#57: RESHAPE
    op_copy(*(ptensors + 138), *(ptensors + 139), 23936);

    // Op#58: ADD
    op_add(*(ptensors + 139), *(ptensors + 49), *(ptensors + 140), 23936,
           1.9571607481e-05f, 0, 4.9245376431e-05f, -6, 3.7824407627e-05f, 6);

    // Op#59: TRANSPOSE
    op_transpose_4d(*(ptensors + 140), *(ptensors + 141), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#60: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 137), *(ptensors + 141), *(ptensors + 142),
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 3.7824407627e-05f, 6, 3.7824371248e-05f, 6);

    // Op#61: TRANSPOSE
    op_transpose_4d(*(ptensors + 142), *(ptensors + 143), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#62: RESHAPE
    op_copy(*(ptensors + 143), *(ptensors + 144), 23936);

    // Op#63: FULLY_CONNECTED
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=3.9634574205e-04f;
    op_fc(*(ptensors + 144), 187, 128, 1,
          (const int8_t*)*(ptensors + 48), (const int32_t*)*(ptensors + 47), *(ptensors + 145),
          3.7824371248e-05f, 6, pws, 3.5996966064e-02f, 20);
      free(pws); }

    // Op#64: ADD
    op_add(*(ptensors + 145), *(ptensors + 120), *(ptensors + 146), 187,
           3.5996966064e-02f, 20, 3.5996969789e-02f, 20, 3.5996966064e-02f, 20);

    // Op#65: SQUARED_DIFFERENCE
    memset(*(ptensors + 147), -128, 187);

    // Op#66: ADD
    op_add(*(ptensors + 147), *(ptensors + 81), *(ptensors + 148), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#67: SUB
    memset(*(ptensors + 149), 0, 187);

    // Op#68: RSQRT
    op_rsqrt(*(ptensors + 148), *(ptensors + 150), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#69: MUL
    op_mul(*(ptensors + 149), *(ptensors + 150), *(ptensors + 151), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#70: EXPAND_DIMS
    op_copy(*(ptensors + 151), *(ptensors + 152), 187);

    // Op#71: CONV_2D
    op_fc(*(ptensors + 152), 187, 1, 4,
          (const int8_t*)*(ptensors + 46), (const int32_t*)*(ptensors + 65), *(ptensors + 153),
          7.8431368067e-09f, 0, pscales_t46, 7.8431368067e-09f, 0);

    // Op#72: RESHAPE
    op_copy(*(ptensors + 153), *(ptensors + 154), 748);

    // Op#73: EXPAND_DIMS
    op_copy(*(ptensors + 154), *(ptensors + 155), 748);

    // Op#74: CONV_2D
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=4.4033546001e-03f;
    op_fc(*(ptensors + 155), 187, 4, 1,
          (const int8_t*)*(ptensors + 45), (const int32_t*)*(ptensors + 60), *(ptensors + 156),
          7.8431368067e-09f, 0, pws, 7.8431368067e-09f, 0);
      free(pws); }

    // Op#75: RESHAPE
    op_copy(*(ptensors + 156), *(ptensors + 157), 187);

    // Op#76: ADD
    op_add(*(ptensors + 157), *(ptensors + 44), *(ptensors + 158), 187,
           7.8431368067e-09f, 0, 3.4954820876e-04f, -128, 3.5996966064e-02f, 17);

    // Op#77: ADD
    op_add(*(ptensors + 158), *(ptensors + 146), *(ptensors + 159), 187,
           3.5996966064e-02f, 17, 3.5996966064e-02f, 20, 3.5996966064e-02f, 17);

    // Op#78: SQUARED_DIFFERENCE
    memset(*(ptensors + 160), -128, 187);

    // Op#79: ADD
    op_add(*(ptensors + 160), *(ptensors + 81), *(ptensors + 161), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#80: SUB
    memset(*(ptensors + 162), 0, 187);

    // Op#81: RSQRT
    op_rsqrt(*(ptensors + 161), *(ptensors + 163), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#82: MUL
    op_mul(*(ptensors + 162), *(ptensors + 163), *(ptensors + 164), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#83: FULLY_CONNECTED
    op_fc(*(ptensors + 164), 187, 1, 128,
          (const int8_t*)*(ptensors + 43), (const int32_t*)*(ptensors + 42), *(ptensors + 165),
          7.8431368067e-09f, 0, pscales_t43, 8.5654037321e-06f, -1);

    // Op#84: RESHAPE
    op_copy(*(ptensors + 165), *(ptensors + 166), 23936);

    // Op#85: ADD
    op_add(*(ptensors + 166), *(ptensors + 41), *(ptensors + 167), 23936,
           8.5654037321e-06f, -1, 3.9215684033e-09f, 0, 8.5653437054e-06f, -1);

    // Op#86: TRANSPOSE
    op_transpose_4d(*(ptensors + 167), *(ptensors + 168), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#87: FULLY_CONNECTED
    op_fc(*(ptensors + 164), 187, 1, 128,
          (const int8_t*)*(ptensors + 40), (const int32_t*)*(ptensors + 39), *(ptensors + 169),
          7.8431368067e-09f, 0, pscales_t40, 8.8167744252e-06f, -2);

    // Op#88: RESHAPE
    op_copy(*(ptensors + 169), *(ptensors + 170), 23936);

    // Op#89: ADD
    op_add(*(ptensors + 170), *(ptensors + 38), *(ptensors + 171), 23936,
           8.8167744252e-06f, -2, 8.5421326901e-08f, -1, 8.8170627350e-06f, -2);

    // Op#90: MUL
    op_mul(*(ptensors + 171), *(ptensors + 74), *(ptensors + 172), 23936,
           8.8170627350e-06f, -2, 9.8039221484e-04f, -128, 2.2042656838e-06f, -2);

    // Op#91: TRANSPOSE
    op_transpose_4d(*(ptensors + 172), *(ptensors + 173), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#92: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 168), *(ptensors + 173), *(ptensors + 174),
                    1, 8, 187, 16,
                    8.5653437054e-06f, -1, 2.2042656838e-06f, -2, 4.2088466046e-09f, -73);

    // Op#93: TRANSPOSE
    op_transpose_4d(*(ptensors + 174), *(ptensors + 175), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#94: SOFTMAX
    op_softmax(*(ptensors + 175), *(ptensors + 176), 1496, 187,
               4.2088466046e-09f, -73, 3.9062500000e-03f, -128);

    // Op#95: FULLY_CONNECTED
    op_fc(*(ptensors + 164), 187, 1, 128,
          (const int8_t*)*(ptensors + 37), (const int32_t*)*(ptensors + 36), *(ptensors + 177),
          7.8431368067e-09f, 0, pscales_t37, 8.8702890935e-06f, -2);

    // Op#96: RESHAPE
    op_copy(*(ptensors + 177), *(ptensors + 178), 23936);

    // Op#97: ADD
    op_add(*(ptensors + 178), *(ptensors + 35), *(ptensors + 179), 23936,
           8.8702890935e-06f, -2, 4.1749022785e-05f, -8, 4.3648789870e-05f, 0);

    // Op#98: TRANSPOSE
    op_transpose_4d(*(ptensors + 179), *(ptensors + 180), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#99: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 176), *(ptensors + 180), *(ptensors + 181),
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 4.3648789870e-05f, 0, 4.3648738938e-05f, 0);

    // Op#100: TRANSPOSE
    op_transpose_4d(*(ptensors + 181), *(ptensors + 182), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#101: RESHAPE
    op_copy(*(ptensors + 182), *(ptensors + 183), 23936);

    // Op#102: FULLY_CONNECTED
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=2.1708158602e-04f;
    op_fc(*(ptensors + 183), 187, 128, 1,
          (const int8_t*)*(ptensors + 34), (const int32_t*)*(ptensors + 33), *(ptensors + 184),
          4.3648738938e-05f, 0, pws, 3.5996966064e-02f, 17);
      free(pws); }

    // Op#103: ADD
    op_add(*(ptensors + 184), *(ptensors + 159), *(ptensors + 185), 187,
           3.5996966064e-02f, 17, 3.5996966064e-02f, 17, 3.5996966064e-02f, 17);

    // Op#104: SQUARED_DIFFERENCE
    memset(*(ptensors + 186), -128, 187);

    // Op#105: ADD
    op_add(*(ptensors + 186), *(ptensors + 81), *(ptensors + 187), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#106: SUB
    memset(*(ptensors + 188), 0, 187);

    // Op#107: RSQRT
    op_rsqrt(*(ptensors + 187), *(ptensors + 189), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#108: MUL
    op_mul(*(ptensors + 188), *(ptensors + 189), *(ptensors + 190), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#109: EXPAND_DIMS
    op_copy(*(ptensors + 190), *(ptensors + 191), 187);

    // Op#110: CONV_2D
    op_fc(*(ptensors + 191), 187, 1, 4,
          (const int8_t*)*(ptensors + 32), (const int32_t*)*(ptensors + 66), *(ptensors + 192),
          7.8431368067e-09f, 0, pscales_t32, 7.8431368067e-09f, 0);

    // Op#111: RESHAPE
    op_copy(*(ptensors + 192), *(ptensors + 193), 748);

    // Op#112: EXPAND_DIMS
    op_copy(*(ptensors + 193), *(ptensors + 194), 748);

    // Op#113: CONV_2D
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=3.3950232901e-03f;
    op_fc(*(ptensors + 194), 187, 4, 1,
          (const int8_t*)*(ptensors + 31), (const int32_t*)*(ptensors + 61), *(ptensors + 195),
          7.8431368067e-09f, 0, pws, 7.8431368067e-09f, 0);
      free(pws); }

    // Op#114: RESHAPE
    op_copy(*(ptensors + 195), *(ptensors + 196), 187);

    // Op#115: ADD
    op_add(*(ptensors + 196), *(ptensors + 30), *(ptensors + 197), 187,
           7.8431368067e-09f, 0, 3.4954832518e-04f, -128, 3.5996966064e-02f, 15);

    // Op#116: ADD
    op_add(*(ptensors + 197), *(ptensors + 185), *(ptensors + 198), 187,
           3.5996966064e-02f, 15, 3.5996966064e-02f, 17, 3.5996966064e-02f, 15);

    // Op#117: SQUARED_DIFFERENCE
    memset(*(ptensors + 199), -128, 187);

    // Op#118: ADD
    op_add(*(ptensors + 199), *(ptensors + 81), *(ptensors + 200), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#119: SUB
    memset(*(ptensors + 201), 0, 187);

    // Op#120: RSQRT
    op_rsqrt(*(ptensors + 200), *(ptensors + 202), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#121: MUL
    op_mul(*(ptensors + 201), *(ptensors + 202), *(ptensors + 203), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#122: FULLY_CONNECTED
    op_fc(*(ptensors + 203), 187, 1, 128,
          (const int8_t*)*(ptensors + 29), (const int32_t*)*(ptensors + 28), *(ptensors + 204),
          7.8431368067e-09f, 0, pscales_t29, 4.5233937840e-07f, -2);

    // Op#123: RESHAPE
    op_copy(*(ptensors + 204), *(ptensors + 205), 23936);

    // Op#124: ADD
    op_add(*(ptensors + 205), *(ptensors + 27), *(ptensors + 206), 23936,
           4.5233937840e-07f, -2, 3.9215684033e-09f, 0, 4.5233127821e-07f, -2);

    // Op#125: TRANSPOSE
    op_transpose_4d(*(ptensors + 206), *(ptensors + 207), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#126: FULLY_CONNECTED
    op_fc(*(ptensors + 203), 187, 1, 128,
          (const int8_t*)*(ptensors + 26), (const int32_t*)*(ptensors + 25), *(ptensors + 208),
          7.8431368067e-09f, 0, pscales_t26, 4.5768948098e-07f, -1);

    // Op#127: RESHAPE
    op_copy(*(ptensors + 208), *(ptensors + 209), 23936);

    // Op#128: ADD
    op_add(*(ptensors + 209), *(ptensors + 24), *(ptensors + 210), 23936,
           4.5768948098e-07f, -1, 1.2812115813e-07f, 5, 4.8657540219e-07f, 1);

    // Op#129: MUL
    op_mul(*(ptensors + 210), *(ptensors + 74), *(ptensors + 211), 23936,
           4.8657540219e-07f, 1, 9.8039221484e-04f, -128, 1.2164385055e-07f, 1);

    // Op#130: TRANSPOSE
    op_transpose_4d(*(ptensors + 211), *(ptensors + 212), 1, 187, 8, 16, 0, 2, 3, 1);

    // Op#131: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 207), *(ptensors + 212), *(ptensors + 213),
                    1, 8, 187, 16,
                    4.5233127821e-07f, -2, 1.2164385055e-07f, 1, 7.8685573612e-09f, 0);

    // Op#132: TRANSPOSE
    op_transpose_4d(*(ptensors + 213), *(ptensors + 214), 1, 8, 187, 187, 0, 1, 3, 2);

    // Op#133: SOFTMAX
    op_softmax(*(ptensors + 214), *(ptensors + 215), 1496, 187,
               7.8685573612e-09f, 0, 3.9062500000e-03f, -128);

    // Op#134: FULLY_CONNECTED
    op_fc(*(ptensors + 203), 187, 1, 128,
          (const int8_t*)*(ptensors + 23), (const int32_t*)*(ptensors + 22), *(ptensors + 216),
          7.8431368067e-09f, 0, pscales_t23, 4.5547096761e-07f, -7);

    // Op#135: RESHAPE
    op_copy(*(ptensors + 216), *(ptensors + 217), 23936);

    // Op#136: ADD
    op_add(*(ptensors + 217), *(ptensors + 21), *(ptensors + 218), 23936,
           4.5547096761e-07f, -7, 2.9470964364e-05f, -5, 2.9520904718e-05f, -4);

    // Op#137: TRANSPOSE
    op_transpose_4d(*(ptensors + 218), *(ptensors + 219), 1, 187, 8, 16, 0, 2, 1, 3);

    // Op#138: BATCH_MATMUL
    op_batch_matmul(*(ptensors + 215), *(ptensors + 219), *(ptensors + 220),
                    1, 8, 187, 187,
                    3.9062500000e-03f, -128, 2.9520904718e-05f, -4, 2.9520893804e-05f, -4);

    // Op#139: TRANSPOSE
    op_transpose_4d(*(ptensors + 220), *(ptensors + 221), 1, 8, 187, 16, 0, 2, 1, 3);

    // Op#140: RESHAPE
    op_copy(*(ptensors + 221), *(ptensors + 222), 23936);

    // Op#141: FULLY_CONNECTED
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=2.9882427771e-04f;
    op_fc(*(ptensors + 222), 187, 128, 1,
          (const int8_t*)*(ptensors + 20), (const int32_t*)*(ptensors + 19), *(ptensors + 223),
          2.9520893804e-05f, -4, pws, 3.5996966064e-02f, 15);
      free(pws); }

    // Op#142: ADD
    op_add(*(ptensors + 223), *(ptensors + 198), *(ptensors + 224), 187,
           3.5996966064e-02f, 15, 3.5996966064e-02f, 15, 3.5996966064e-02f, 15);

    // Op#143: SQUARED_DIFFERENCE
    memset(*(ptensors + 225), -128, 187);

    // Op#144: ADD
    op_add(*(ptensors + 225), *(ptensors + 81), *(ptensors + 226), 187,
           3.9215688048e-06f, -128, 3.9215688048e-06f, -128, 3.9215688048e-06f, -128);

    // Op#145: SUB
    memset(*(ptensors + 227), 0, 187);

    // Op#146: RSQRT
    op_rsqrt(*(ptensors + 226), *(ptensors + 228), 187,
             3.9215688048e-06f, -128, 1.2401087582e-01f, -128);

    // Op#147: MUL
    op_mul(*(ptensors + 227), *(ptensors + 228), *(ptensors + 229), 187,
           7.8431368067e-09f, 0, 1.2401087582e-01f, -128, 7.8431368067e-09f, 0);

    // Op#148: EXPAND_DIMS
    op_copy(*(ptensors + 229), *(ptensors + 230), 187);

    // Op#149: CONV_2D
    op_fc(*(ptensors + 230), 187, 1, 4,
          (const int8_t*)*(ptensors + 18), (const int32_t*)*(ptensors + 67), *(ptensors + 231),
          7.8431368067e-09f, 0, pscales_t18, 7.8431368067e-09f, 0);

    // Op#150: RESHAPE
    op_copy(*(ptensors + 231), *(ptensors + 232), 748);

    // Op#151: EXPAND_DIMS
    op_copy(*(ptensors + 232), *(ptensors + 233), 748);

    // Op#152: CONV_2D
    { float* pws = (float*)malloc(1 << 2);
      for(volatile int t0=0; t0<1; t0++) *(pws+t0)=8.1268763170e-03f;
    op_fc(*(ptensors + 233), 187, 4, 1,
          (const int8_t*)*(ptensors + 17), (const int32_t*)*(ptensors + 62), *(ptensors + 234),
          7.8431368067e-09f, 0, pws, 7.8431368067e-09f, 0);
      free(pws); }

    // Op#153: RESHAPE
    op_copy(*(ptensors + 234), *(ptensors + 235), 187);

    // Op#154: ADD
    op_add(*(ptensors + 235), *(ptensors + 16), *(ptensors + 236), 187,
           7.8431368067e-09f, 0, 3.4954876173e-04f, -128, 3.5996966064e-02f, 12);

    // Op#155: ADD
    op_add(*(ptensors + 236), *(ptensors + 224), *(ptensors + 237), 187,
           3.5996966064e-02f, 12, 3.5996966064e-02f, 15, 3.5996966064e-02f, 12);

    // Op#156: MEAN
    op_mean(*(ptensors + 237), *(ptensors + 238),
            187, 1, 1,
            3.5996966064e-02f, 12, 3.5996966064e-02f, 12);

    // Op#157: FULLY_CONNECTED
    op_fc(*(ptensors + 238), 1, 187, 128,
          (const int8_t*)*(ptensors + 15), (const int32_t*)*(ptensors + 14), *(ptensors + 239),
          3.5996966064e-02f, 12, pscales_t15, 2.6135431603e-02f, -128);

    // Op#158: FULLY_CONNECTED
    op_fc(*(ptensors + 239), 1, 128, 64,
          (const int8_t*)*(ptensors + 13), (const int32_t*)*(ptensors + 12), *(ptensors + 240),
          2.6135431603e-02f, -128, pscales_t13, 3.1560700387e-02f, -128);

    // Op#159: FULLY_CONNECTED
    op_fc(*(ptensors + 240), 1, 64, 5,
          (const int8_t*)*(ptensors + 11), (const int32_t*)*(ptensors + 10), *(ptensors + 241),
          3.1560700387e-02f, -128, pscales_t11, 1.9118081033e-01f, 14);

    // Op#160: SOFTMAX
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
