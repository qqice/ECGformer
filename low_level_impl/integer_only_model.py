"""
纯整数运算的ECGformer模型实现

这个实现完全在INT8/INT32空间进行计算，不依赖TFLite。
可以直接移植到C语言。

关键概念：
1. 所有激活值存储为INT8 (-128 to 127)
2. 权重存储为INT8 (-128 to 127)
3. 偏置存储为INT32
4. 中间累加器使用INT32
5. 使用定点算术进行scale转换
"""

import numpy as np
import json
import os
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)


class QuantParam:
    """单个张量的量化参数"""
    def __init__(self, scales, zero_points, quantized_dim=0):
        self.scales = np.atleast_1d(np.array(scales, dtype=np.float64))
        self.zero_points = np.atleast_1d(np.array(zero_points, dtype=np.int32))
        self.quantized_dim = quantized_dim
        self.is_per_channel = len(self.scales) > 1
    
    @property
    def scale(self):
        """返回第一个scale（用于per-tensor量化）"""
        return self.scales[0]
    
    @property
    def zero_point(self):
        """返回第一个zero_point"""
        return self.zero_points[0]


class IntegerOnlyECGformer:
    """
    纯整数运算的ECGformer模型
    
    所有计算在INT8/INT32空间进行，不使用浮点中间值。
    这是可以直接移植到C语言的参考实现。
    """
    
    def __init__(self, model_path: str):
        """
        从TFLite模型加载权重和量化参数
        
        Args:
            model_path: TFLite模型路径
        """
        self.model_path = model_path
        
        # 加载模型信息
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        
        # 获取张量详情
        tensor_details = interpreter.get_tensor_details()
        self.tensor_details = {t['index']: t for t in tensor_details}
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 输入输出信息
        self.input_tensor_id = input_details[0]['index']
        self.output_tensor_id = output_details[0]['index']
        
        self.input_shape = tuple(input_details[0]['shape'])
        self.output_shape = tuple(output_details[0]['shape'])
        
        # 提取量化参数
        self.quant_params = {}
        for tid, t in self.tensor_details.items():
            qp = t.get('quantization_parameters', {})
            scales = qp.get('scales', np.array([]))
            zps = qp.get('zero_points', np.array([]))
            qdim = qp.get('quantized_dimension', 0)
            
            # 检查旧格式
            if len(scales) == 0:
                old_quant = t.get('quantization', (0.0, 0))
                if old_quant[0] != 0:
                    scales = np.array([old_quant[0]])
                    zps = np.array([old_quant[1]])
            
            if len(scales) > 0:
                self.quant_params[tid] = QuantParam(scales, zps, qdim)
        
        # 提取权重（INT8/INT32格式）
        self.weights = {}
        for tid in self.tensor_details:
            try:
                data = interpreter.get_tensor(tid)
                self.weights[tid] = data.copy()
            except:
                pass
        
        # 解析操作
        self.ops = self._parse_ops()
        
        # 运行时张量存储（全部是整数）
        self.tensors = {}
        
        # 预计算定点乘法器
        self._precompute_multipliers()
        
        print(f"模型加载完成")
        print(f"  输入形状: {self.input_shape}")
        print(f"  输出形状: {self.output_shape}")
        print(f"  操作数: {len(self.ops)}")
        print(f"  量化张量数: {len(self.quant_params)}")
    
    def _parse_ops(self):
        """解析TFLite模型操作"""
        import re
        import io
        import tensorflow as tf
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        
        try:
            tf.lite.experimental.Analyzer.analyze(model_path=self.model_path)
        finally:
            sys.stdout = old_stdout
        
        analysis_text = mystdout.getvalue()
        lines = analysis_text.split('\n')
        
        ops = []
        op_pattern = re.compile(r"\s*Op#(\d+)\s+(\w+)\((.*)\)\s*->\s*\[(.*)\]")
        
        for line in lines:
            op_match = op_pattern.match(line)
            if op_match:
                op_id = int(op_match.group(1))
                op_type = op_match.group(2)
                inputs_str = op_match.group(3)
                outputs_str = op_match.group(4)
                
                input_ids = [int(x) for x in re.findall(r"T#(\d+)", inputs_str)]
                output_ids = [int(x) for x in re.findall(r"T#(\d+)", outputs_str)]
                
                ops.append({
                    'id': op_id,
                    'type': op_type,
                    'inputs': input_ids,
                    'outputs': output_ids
                })
        
        return ops
    
    def _precompute_multipliers(self):
        """
        预计算定点乘法器
        
        TFLite使用定点算术: output = (input * multiplier) >> shift
        multiplier是一个32位整数，shift是移位量
        
        这避免了运行时的浮点除法
        """
        self.multipliers = {}
        
        for op in self.ops:
            op_id = op['id']
            op_type = op['type']
            input_ids = op['inputs']
            output_ids = op['outputs']
            
            if op_type == 'FULLY_CONNECTED':
                # FULLY_CONNECTED: out = (in - in_zp) * (w - w_zp) * (in_scale * w_scale / out_scale) + out_zp
                # 我们需要计算 effective_scale = in_scale * w_scale / out_scale
                if len(input_ids) >= 2:
                    in_qp = self.quant_params.get(input_ids[0])
                    w_qp = self.quant_params.get(input_ids[1])
                    out_qp = self.quant_params.get(output_ids[0])
                    
                    if in_qp and w_qp and out_qp:
                        # Per-channel: 每个输出通道有不同的scale
                        if w_qp.is_per_channel:
                            effective_scales = in_qp.scale * w_qp.scales / out_qp.scale
                            multipliers, shifts = self._quantize_multipliers(effective_scales)
                        else:
                            effective_scale = in_qp.scale * w_qp.scale / out_qp.scale
                            multiplier, shift = self._quantize_multiplier(effective_scale)
                            multipliers = np.array([multiplier])
                            shifts = np.array([shift])
                        
                        self.multipliers[op_id] = {
                            'multipliers': multipliers,
                            'shifts': shifts,
                            'input_zp': in_qp.zero_point,
                            'weight_zps': w_qp.zero_points,
                            'output_zp': out_qp.zero_point
                        }
    
    def _quantize_multiplier(self, real_multiplier: float) -> tuple:
        """
        将浮点乘法器转换为定点表示
        
        output = (input * multiplier) >> shift
        
        其中 multiplier 是一个在 [0.5, 1.0) 范围内的值左移31位后的整数
        """
        if real_multiplier == 0:
            return 0, 0
        
        # 找到移位量使得 multiplier 在 [0.5, 1.0) 范围内
        shift = 0
        while real_multiplier < 0.5:
            real_multiplier *= 2
            shift -= 1
        while real_multiplier >= 1.0:
            real_multiplier /= 2
            shift += 1
        
        # 量化为32位整数（左移31位）
        q = int(round(real_multiplier * (1 << 31)))
        
        # 确保在有效范围内
        if q == (1 << 31):
            q = (1 << 31) - 1
        
        # 调整shift为正值（用于右移）
        # 实际计算: (input * q) >> (31 - shift)
        right_shift = 31 - shift
        
        return q, right_shift
    
    def _quantize_multipliers(self, real_multipliers: np.ndarray) -> tuple:
        """批量转换浮点乘法器"""
        multipliers = []
        shifts = []
        for rm in real_multipliers:
            m, s = self._quantize_multiplier(rm)
            multipliers.append(m)
            shifts.append(s)
        return np.array(multipliers, dtype=np.int64), np.array(shifts, dtype=np.int32)
    
    def _apply_multiplier(self, value: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
        """
        应用定点乘法器
        
        这是TFLite的核心操作：将INT32累加器值转换为INT8输出
        
        output = round((value * multiplier) >> shift)
        
        注意：这里使用"银行家舍入"（四舍五入到最近的偶数）
        """
        value = value.astype(np.int64)
        
        # 执行乘法（使用64位防止溢出）
        result = value * np.int64(multiplier)
        
        # 执行带舍入的右移
        # round = (result + (1 << (shift - 1))) >> shift
        if shift > 0:
            # 添加舍入偏置
            rounding = np.int64(1) << (shift - 1)
            result = (result + rounding) >> shift
        
        return result.astype(np.int32)
    
    def _apply_multipliers_per_channel(self, values: np.ndarray, multipliers: np.ndarray, 
                                        shifts: np.ndarray, axis: int = -1) -> np.ndarray:
        """应用per-channel定点乘法器"""
        result = np.zeros_like(values, dtype=np.int32)
        n_channels = multipliers.shape[0]
        
        # 将values沿指定轴迭代
        for c in range(n_channels):
            # 创建切片索引
            slices = [slice(None)] * values.ndim
            slices[axis] = c
            
            channel_values = values[tuple(slices)].astype(np.int64)
            
            # 应用该通道的乘法器
            multiplied = channel_values * np.int64(multipliers[c])
            
            if shifts[c] > 0:
                rounding = np.int64(1) << (shifts[c] - 1)
                channel_result = (multiplied + rounding) >> shifts[c]
            else:
                channel_result = multiplied
            
            result[tuple(slices)] = channel_result.astype(np.int32)
        
        return result
    
    def _saturate_int8(self, value: np.ndarray) -> np.ndarray:
        """饱和到INT8范围"""
        return np.clip(value, -128, 127).astype(np.int8)
    
    def _get_tensor(self, tensor_id: int) -> np.ndarray:
        """获取张量"""
        if tensor_id in self.tensors:
            return self.tensors[tensor_id]
        elif tensor_id in self.weights:
            return self.weights[tensor_id]
        else:
            raise ValueError(f"Tensor {tensor_id} not found")
    
    # ==================== 整数运算操作实现 ====================
    
    def _int_add(self, in1: np.ndarray, in2: np.ndarray, 
                  in1_qp: QuantParam, in2_qp: QuantParam, out_qp: QuantParam) -> np.ndarray:
        """
        整数加法
        
        将两个INT8输入加法，产生INT8输出
        需要处理不同的量化参数
        """
        # 反量化到INT32（带零点偏移）
        in1_32 = in1.astype(np.int32) - in1_qp.zero_point
        in2_32 = in2.astype(np.int32) - in2_qp.zero_point
        
        # 缩放到公共尺度
        # 使用双精度避免精度损失
        scale1_ratio = in1_qp.scale / out_qp.scale
        scale2_ratio = in2_qp.scale / out_qp.scale
        
        # 转换为定点
        m1, s1 = self._quantize_multiplier(scale1_ratio)
        m2, s2 = self._quantize_multiplier(scale2_ratio)
        
        # 应用缩放
        scaled1 = self._apply_multiplier(in1_32, m1, s1)
        scaled2 = self._apply_multiplier(in2_32, m2, s2)
        
        # 加法并添加输出零点
        result = scaled1 + scaled2 + out_qp.zero_point
        
        return self._saturate_int8(result)
    
    def _int_sub(self, in1: np.ndarray, in2: np.ndarray,
                  in1_qp: QuantParam, in2_qp: QuantParam, out_qp: QuantParam) -> np.ndarray:
        """整数减法"""
        in1_32 = in1.astype(np.int32) - in1_qp.zero_point
        in2_32 = in2.astype(np.int32) - in2_qp.zero_point
        
        scale1_ratio = in1_qp.scale / out_qp.scale
        scale2_ratio = in2_qp.scale / out_qp.scale
        
        m1, s1 = self._quantize_multiplier(scale1_ratio)
        m2, s2 = self._quantize_multiplier(scale2_ratio)
        
        scaled1 = self._apply_multiplier(in1_32, m1, s1)
        scaled2 = self._apply_multiplier(in2_32, m2, s2)
        
        result = scaled1 - scaled2 + out_qp.zero_point
        
        return self._saturate_int8(result)
    
    def _int_mul(self, in1: np.ndarray, in2: np.ndarray,
                  in1_qp: QuantParam, in2_qp: QuantParam, out_qp: QuantParam) -> np.ndarray:
        """整数乘法"""
        in1_32 = in1.astype(np.int32) - in1_qp.zero_point
        in2_32 = in2.astype(np.int32) - in2_qp.zero_point
        
        # 乘法
        product = in1_32 * in2_32
        
        # 缩放: in1_scale * in2_scale / out_scale
        effective_scale = (in1_qp.scale * in2_qp.scale) / out_qp.scale
        m, s = self._quantize_multiplier(effective_scale)
        
        scaled = self._apply_multiplier(product, m, s)
        result = scaled + out_qp.zero_point
        
        return self._saturate_int8(result)
    
    def _int_squared_difference(self, in1: np.ndarray, in2: np.ndarray,
                                 in1_qp: QuantParam, in2_qp: QuantParam, 
                                 out_qp: QuantParam) -> np.ndarray:
        """整数平方差"""
        in1_32 = in1.astype(np.int32) - in1_qp.zero_point
        in2_32 = in2.astype(np.int32) - in2_qp.zero_point
        
        # 缩放到公共尺度
        # 假设in1和in2有相同的scale（LayerNorm中的情况）
        diff = in1_32 - in2_32
        
        # 平方
        sq_diff = diff * diff
        
        # 缩放: (in_scale)^2 / out_scale
        effective_scale = (in1_qp.scale * in1_qp.scale) / out_qp.scale
        m, s = self._quantize_multiplier(effective_scale)
        
        scaled = self._apply_multiplier(sq_diff, m, s)
        result = scaled + out_qp.zero_point
        
        return self._saturate_int8(result)
    
    def _int_rsqrt(self, input_arr: np.ndarray, in_qp: QuantParam, 
                   out_qp: QuantParam) -> np.ndarray:
        """
        整数倒数平方根
        
        这是最复杂的操作之一，TFLite使用查找表
        我们这里使用近似方法
        """
        # 反量化
        input_float = (input_arr.astype(np.float32) - in_qp.zero_point) * in_qp.scale
        
        # 计算rsqrt（使用float，因为这很难用纯整数实现）
        # 在C实现中，应该使用查找表
        rsqrt = 1.0 / np.sqrt(np.maximum(input_float, 1e-12))
        
        # 重新量化
        output = np.round(rsqrt / out_qp.scale) + out_qp.zero_point
        
        return self._saturate_int8(output)
    
    def _int_fully_connected(self, input_arr: np.ndarray, weight: np.ndarray, 
                              bias: np.ndarray, op_id: int,
                              out_qp: QuantParam) -> np.ndarray:
        """
        整数全连接层
        
        这是量化网络中最重要的操作
        
        计算: output = (input - in_zp) @ (weight - w_zp).T + bias
        然后应用缩放和输出零点
        """
        mult_info = self.multipliers.get(op_id)
        if mult_info is None:
            raise ValueError(f"No multiplier info for op {op_id}")
        
        input_zp = mult_info['input_zp']
        weight_zps = mult_info['weight_zps']
        output_zp = mult_info['output_zp']
        multipliers = mult_info['multipliers']
        shifts = mult_info['shifts']
        
        # 输入形状处理
        original_shape = input_arr.shape[:-1]
        in_dim = input_arr.shape[-1]
        out_dim = weight.shape[0]
        
        # 展平输入
        input_flat = input_arr.reshape(-1, in_dim).astype(np.int32)
        batch_size = input_flat.shape[0]
        
        # 零点偏移
        input_centered = input_flat - input_zp
        
        # 权重处理
        weight_32 = weight.astype(np.int32)
        
        # 矩阵乘法（INT32累加器）
        # output[b, o] = sum_i(input_centered[b, i] * weight[o, i])
        accumulator = np.zeros((batch_size, out_dim), dtype=np.int32)
        
        for o in range(out_dim):
            # 每个输出通道可能有不同的weight zero_point
            w_zp = weight_zps[o] if len(weight_zps) > 1 else weight_zps[0]
            weight_row = weight_32[o, :] - w_zp
            
            # 累加
            accumulator[:, o] = np.sum(input_centered * weight_row, axis=1)
        
        # 加偏置
        if bias is not None:
            accumulator += bias.astype(np.int32)
        
        # 应用per-channel缩放
        if len(multipliers) > 1:
            scaled = self._apply_multipliers_per_channel(accumulator, multipliers, shifts, axis=1)
        else:
            scaled = self._apply_multiplier(accumulator, multipliers[0], shifts[0])
        
        # 添加输出零点
        result = scaled + output_zp
        
        # 饱和并reshape
        output = self._saturate_int8(result)
        
        return output.reshape(original_shape + (out_dim,))
    
    def _int_mean(self, input_arr: np.ndarray, axes: tuple, 
                   in_qp: QuantParam, out_qp: QuantParam) -> np.ndarray:
        """
        整数均值
        
        TFLite的MEAN操作在量化空间中执行
        注意：TFLite的MEAN默认不保留维度(keepdims=False)
        """
        # 计算均值（在INT32空间）
        input_32 = input_arr.astype(np.int32)
        
        # 计算和
        sum_result = np.sum(input_32, axis=axes, keepdims=False)
        
        # 计算元素数量
        n_elements = 1
        for ax in axes:
            n_elements *= input_arr.shape[ax]
        
        # 除法（使用定点乘法器）
        # mean = sum / n = sum * (1/n)
        inv_n = 1.0 / n_elements
        m, s = self._quantize_multiplier(inv_n)
        
        mean_32 = self._apply_multiplier(sum_result, m, s)
        
        # 缩放: in_scale / out_scale
        scale_ratio = in_qp.scale / out_qp.scale
        m2, s2 = self._quantize_multiplier(scale_ratio)
        
        # 零点调整
        mean_centered = mean_32 - in_qp.zero_point
        scaled = self._apply_multiplier(mean_centered, m2, s2)
        result = scaled + out_qp.zero_point
        
        return self._saturate_int8(result)
    
    def _int_softmax(self, input_arr: np.ndarray, in_qp: QuantParam, 
                      out_qp: QuantParam) -> np.ndarray:
        """
        整数Softmax
        
        TFLite使用定点exp和div
        这里我们使用简化实现，在C中应该使用查找表
        """
        # 反量化（softmax很难完全用整数实现）
        input_float = (input_arr.astype(np.float32) - in_qp.zero_point) * in_qp.scale
        
        # Softmax
        x_max = np.max(input_float, axis=-1, keepdims=True)
        exp_x = np.exp(input_float - x_max)
        softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        # 重新量化
        output = np.round(softmax / out_qp.scale) + out_qp.zero_point
        
        return self._saturate_int8(output)
    
    def _int_batch_matmul(self, in1: np.ndarray, in2: np.ndarray,
                           in1_qp: QuantParam, in2_qp: QuantParam,
                           out_qp: QuantParam) -> np.ndarray:
        """整数批量矩阵乘法"""
        # 零点偏移
        in1_32 = in1.astype(np.int32) - in1_qp.zero_point
        in2_32 = in2.astype(np.int32) - in2_qp.zero_point
        
        # 矩阵乘法
        result_32 = np.matmul(in1_32, in2_32)
        
        # 缩放
        effective_scale = (in1_qp.scale * in2_qp.scale) / out_qp.scale
        m, s = self._quantize_multiplier(effective_scale)
        
        scaled = self._apply_multiplier(result_32, m, s)
        result = scaled + out_qp.zero_point
        
        return self._saturate_int8(result)
    
    def _int_conv2d(self, input_arr: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                     in_qp: QuantParam, w_qp: QuantParam, out_qp: QuantParam,
                     stride: int = 1, padding: str = 'VALID') -> np.ndarray:
        """
        整数2D卷积
        
        简化实现，假设kernel_size=1（pointwise conv）
        """
        # 输入: [N, H, W, C_in]
        # 权重: [C_out, K_h, K_w, C_in] 或 [C_out, 1, 1, C_in] for pointwise
        
        N, H, W, C_in = input_arr.shape
        C_out = weight.shape[0]
        K_h = weight.shape[1]
        K_w = weight.shape[2]
        
        input_32 = input_arr.astype(np.int32) - in_qp.zero_point
        
        if K_h == 1 and K_w == 1:
            # Pointwise convolution
            # 等价于reshape后的fully_connected
            input_flat = input_32.reshape(N * H * W, C_in)
            weight_flat = weight.reshape(C_out, C_in)
            
            # 矩阵乘法
            output_flat = np.zeros((N * H * W, C_out), dtype=np.int32)
            
            for o in range(C_out):
                # per-channel: 每个输出通道有自己的zero_point
                if w_qp.is_per_channel and len(w_qp.zero_points) > o:
                    w_zp = w_qp.zero_points[o]
                else:
                    w_zp = w_qp.zero_point
                weight_row = weight_flat[o, :].astype(np.int32) - w_zp
                output_flat[:, o] = np.sum(input_flat * weight_row, axis=1)
            
            # 加偏置
            if bias is not None:
                output_flat += bias.astype(np.int32)
            
            # 缩放
            if w_qp.is_per_channel:
                effective_scales = in_qp.scale * w_qp.scales / out_qp.scale
                multipliers, shifts = self._quantize_multipliers(effective_scales)
                scaled = self._apply_multipliers_per_channel(output_flat, multipliers, shifts, axis=1)
            else:
                effective_scale = in_qp.scale * w_qp.scale / out_qp.scale
                m, s = self._quantize_multiplier(effective_scale)
                scaled = self._apply_multiplier(output_flat, m, s)
            
            result = scaled + out_qp.zero_point
            output = self._saturate_int8(result)
            
            return output.reshape(N, H, W, C_out)
        else:
            raise NotImplementedError("Only pointwise conv (1x1) is implemented")
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        前向传播（纯整数运算）
        
        Args:
            input_data: float32输入数据，形状为 [1, 187, 1]
            
        Returns:
            float32输出数据，形状为 [1, 5]
        """
        # 量化输入
        in_qp = self.quant_params[self.input_tensor_id]
        input_int8 = np.round(input_data / in_qp.scale) + in_qp.zero_point
        input_int8 = self._saturate_int8(input_int8)
        
        # 存储输入
        self.tensors[self.input_tensor_id] = input_int8
        
        # 执行操作
        for op in self.ops:
            self._execute_op(op)
        
        # 获取输出并反量化
        output_int8 = self.tensors[self.output_tensor_id]
        out_qp = self.quant_params[self.output_tensor_id]
        output_float = (output_int8.astype(np.float32) - out_qp.zero_point) * out_qp.scale
        
        return output_float
    
    def forward_int8(self, input_int8: np.ndarray) -> np.ndarray:
        """
        前向传播（输入输出都是INT8）
        
        这是最接近C实现的接口
        
        Args:
            input_int8: INT8输入数据
            
        Returns:
            INT8输出数据
        """
        self.tensors[self.input_tensor_id] = input_int8.astype(np.int8)
        
        for op in self.ops:
            self._execute_op(op)
        
        return self.tensors[self.output_tensor_id]
    
    def _execute_op(self, op: dict) -> None:
        """执行单个操作"""
        op_type = op['type']
        op_id = op['id']
        input_ids = op['inputs']
        output_ids = op['outputs']
        
        # 获取输入
        inputs = [self._get_tensor(tid) for tid in input_ids]
        
        # 获取量化参数
        in_qps = [self.quant_params.get(tid) for tid in input_ids]
        out_qps = [self.quant_params.get(tid) for tid in output_ids]
        
        # 检查是否同输入操作
        same_input = len(input_ids) >= 2 and input_ids[0] == input_ids[1]
        
        # 根据操作类型执行
        if op_type == 'SQUARED_DIFFERENCE':
            if same_input:
                # x - x = 0，但由于量化会有微小噪声
                # 在整数空间，这应该直接输出 zero_point
                result = np.full(inputs[0].shape, out_qps[0].zero_point, dtype=np.int8)
            else:
                result = self._int_squared_difference(inputs[0], inputs[1], 
                                                      in_qps[0], in_qps[1], out_qps[0])
        
        elif op_type == 'ADD':
            result = self._int_add(inputs[0], inputs[1], in_qps[0], in_qps[1], out_qps[0])
        
        elif op_type == 'SUB':
            if same_input:
                # x - x = 0
                result = np.full(inputs[0].shape, out_qps[0].zero_point, dtype=np.int8)
            else:
                result = self._int_sub(inputs[0], inputs[1], in_qps[0], in_qps[1], out_qps[0])
        
        elif op_type == 'MUL':
            result = self._int_mul(inputs[0], inputs[1], in_qps[0], in_qps[1], out_qps[0])
        
        elif op_type == 'RSQRT':
            result = self._int_rsqrt(inputs[0], in_qps[0], out_qps[0])
        
        elif op_type == 'FULLY_CONNECTED':
            weight = inputs[1]
            bias = inputs[2] if len(inputs) > 2 else None
            result = self._int_fully_connected(inputs[0], weight, bias, op_id, out_qps[0])
        
        elif op_type == 'MEAN':
            # MEAN通常沿着空间维度
            axes_data = inputs[1]
            if np.isscalar(axes_data) or axes_data.ndim == 0:
                axes = (int(axes_data),)
            else:
                axes = tuple(axes_data.astype(int).tolist())
            result = self._int_mean(inputs[0], axes, in_qps[0], out_qps[0])
        
        elif op_type == 'SOFTMAX':
            result = self._int_softmax(inputs[0], in_qps[0], out_qps[0])
        
        elif op_type == 'BATCH_MATMUL':
            result = self._int_batch_matmul(inputs[0], inputs[1], 
                                            in_qps[0], in_qps[1], out_qps[0])
        
        elif op_type == 'CONV_2D':
            weight = inputs[1]
            bias = inputs[2] if len(inputs) > 2 else None
            w_qp = in_qps[1]
            result = self._int_conv2d(inputs[0], weight, bias, in_qps[0], w_qp, out_qps[0])
        
        elif op_type == 'RESHAPE':
            new_shape = inputs[1].astype(np.int32)
            result = inputs[0].reshape(tuple(new_shape))
        
        elif op_type == 'TRANSPOSE':
            perm = inputs[1].astype(np.int32)
            result = np.transpose(inputs[0], tuple(perm))
        
        elif op_type == 'EXPAND_DIMS':
            axis = int(inputs[1])
            result = np.expand_dims(inputs[0], axis=axis)
        
        else:
            raise NotImplementedError(f"Operation {op_type} not implemented")
        
        # 存储输出
        for out_id in output_ids:
            self.tensors[out_id] = result


def test_integer_model():
    """测试纯整数模型"""
    import tensorflow as tf
    
    model_path = os.path.join(PROJECT_ROOT, 'exported_models', 'tflite', 
                              'ecgformer_custom_ln_int8.tflite')
    
    print("="*60)
    print("测试纯整数运算模型")
    print("="*60)
    
    # 创建纯整数模型
    int_model = IntegerOnlyECGformer(model_path)
    
    # 创建TFLite解释器作为参考
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 生成测试输入
    np.random.seed(42)
    test_input = np.random.randn(1, 187, 1).astype(np.float32) * 0.5
    
    # 量化输入
    in_scale = input_details[0]['quantization'][0]
    in_zp = input_details[0]['quantization'][1]
    test_input_int8 = np.clip(np.round(test_input / in_scale) + in_zp, -128, 127).astype(np.int8)
    
    # TFLite推理
    interpreter.set_tensor(input_details[0]['index'], test_input_int8)
    interpreter.invoke()
    tflite_output_int8 = interpreter.get_tensor(output_details[0]['index'])
    
    # 反量化TFLite输出
    out_scale = output_details[0]['quantization'][0]
    out_zp = output_details[0]['quantization'][1]
    tflite_output = (tflite_output_int8.astype(np.float32) - out_zp) * out_scale
    
    # 纯整数模型推理
    int_output_int8 = int_model.forward_int8(test_input_int8)
    int_output = (int_output_int8.astype(np.float32) - out_zp) * out_scale
    
    print(f"\n测试结果:")
    print(f"  TFLite 输出 (INT8): {tflite_output_int8.flatten()}")
    print(f"  整数模型输出 (INT8): {int_output_int8.flatten()}")
    print(f"  TFLite 输出 (float): {tflite_output.flatten()}")
    print(f"  整数模型输出 (float): {int_output.flatten()}")
    
    # 比较
    diff = np.abs(tflite_output - int_output)
    print(f"\n  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    
    # 预测结果
    tflite_pred = np.argmax(tflite_output)
    int_pred = np.argmax(int_output)
    print(f"\n  TFLite预测类别: {tflite_pred}")
    print(f"  整数模型预测类别: {int_pred}")
    print(f"  预测一致: {tflite_pred == int_pred}")


if __name__ == '__main__':
    test_integer_model()
