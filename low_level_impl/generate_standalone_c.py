# -*- coding: utf-8 -*-
"""
生成完整独立的ECGformer C实现代码 - Bare-metal Hardware Verification Style

这个脚本生成一个完整独立的C文件，包含所有权重数据和推理代码。
可以直接编译运行，无需其他头文件。
针对RISC-V AI加速器硬件验证优化。
"""

import numpy as np
import os
import sys
import re
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf


class ECGformerCGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        self.interpreter.allocate_tensors()
        
        self.tensor_details = self.interpreter.get_tensor_details()
        self.tensor_dict = {t['index']: t for t in self.tensor_details}
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 解析操作
        self.ops = self._parse_ops()
        
        # 张量信息和数据
        self.tensors_info = {}
        self.weights_data = {}
        self._load_tensor_info()
        
        # 确定常量和激活张量
        self.constant_tensors = set()
        self.activation_tensors = set()
        self._classify_tensors()
        
        # 张量生命周期分析 (用于内存复用)
        self.tensor_lifetime = {}  # {tid: (first_use_op, last_use_op)}
        self._analyze_tensor_lifetime()
        
    def _parse_ops(self):
        """解析TFLite模型操作"""
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
    
    def _load_tensor_info(self):
        """加载所有张量信息"""
        for t in self.tensor_details:
            tid = t['index']
            shape = tuple(t['shape'])
            size = int(np.prod(shape)) if len(shape) > 0 else 1
            
            # 量化参数
            qp = t.get('quantization_parameters', {})
            scales = qp.get('scales', np.array([]))
            zps = qp.get('zero_points', np.array([]))
            qdim = qp.get('quantized_dimension', 0)
            old_q = t.get('quantization', (0.0, 0))
            
            if len(scales) == 0 and old_q[0] != 0:
                scales = np.array([old_q[0]])
                zps = np.array([old_q[1]])
            
            self.tensors_info[tid] = {
                'shape': shape,
                'size': size,
                'scales': scales,
                'zps': zps,
                'qdim': qdim,
                'dtype': str(t['dtype'])
            }
            
            # 尝试获取权重数据
            try:
                data = self.interpreter.get_tensor(tid)
                self.weights_data[tid] = data.copy()
            except:
                pass
    
    def _classify_tensors(self):
        """分类张量：常量vs激活"""
        output_tensors = set()
        for op in self.ops:
            for tid in op['outputs']:
                output_tensors.add(tid)
        
        # 输入张量ID
        input_tid = self.input_details[0]['index']
        
        for op in self.ops:
            for tid in op['inputs']:
                # 排除输入张量和其他操作的输出
                if tid in self.weights_data and tid not in output_tensors and tid != input_tid:
                    self.constant_tensors.add(tid)
            for tid in op['outputs']:
                self.activation_tensors.add(tid)
        
        # 输入张量也是激活张量
        self.activation_tensors.add(input_tid)
        
        # 从常量张量中移除激活张量
        self.constant_tensors -= self.activation_tensors
        
        # 预扫描收集 per-channel Q16 scales (适配INT26)
        self._precompute_q16_scales()

    def _analyze_tensor_lifetime(self):
        """分析每个激活张量的生命周期
        
        计算每个张量的:
        - first_use: 首次使用的op编号（作为输出创建）
        - last_use: 最后使用的op编号（作为输入被消费）
        """
        output_tid = self.output_details[0]['index']
        input_tid = self.input_details[0]['index']
        
        # first_use: 张量作为输出被创建的位置
        # 对于输入张量，first_use = -1 (在推理开始前就存在)
        first_use = {}
        last_use = {}
        
        # 初始化输入张量
        first_use[input_tid] = -1
        last_use[input_tid] = -1  # 将被后续操作更新
        
        for op in self.ops:
            op_idx = op['id']
            
            # 输出张量在此操作被创建
            for tid in op['outputs']:
                if tid in self.activation_tensors:
                    first_use[tid] = op_idx
            
            # 输入张量在此操作被使用
            for tid in op['inputs']:
                if tid in self.activation_tensors:
                    last_use[tid] = op_idx
        
        # 输出张量需要一直保留到推理结束
        last_use[output_tid] = len(self.ops)
        
        # 构建生命周期字典
        for tid in self.activation_tensors:
            f = first_use.get(tid, 0)
            l = last_use.get(tid, len(self.ops))
            self.tensor_lifetime[tid] = (f, l)
        
        # 打印生命周期统计
        print(f"  张量生命周期分析完成: {len(self.tensor_lifetime)} 个激活张量")
    
    def _compute_pingpong_allocation(self):
        """计算内存分配 (pingpong 模式 - 支持操作重排和注意力融合)
        
        策略:
        1. 识别注意力块并检查是否可以重排 V 的计算
        2. 如果可以重排，将中间注意力张量 (8,187,187) 排除出内存分配
        3. 使用逐头计算，每次只需 ~35KB 临时空间
        """
        limit = self.memory_limit
        
        # 识别注意力相关操作（包含重排分析）
        self._identify_attention_blocks()
        
        # 检查有多少注意力块可以融合
        fusable_blocks = [b for b in self.attention_blocks if b.get('can_reorder', False)]
        
        if fusable_blocks:
            print(f"  发现 {len(fusable_blocks)} 个可融合的注意力块")
        else:
            print(f"  无可融合的注意力块")
        
        # 使用支持融合的内存分配
        self.slot_assignments, self.slot_sizes, self.peak_memory = \
            self._compute_memory_reuse_allocation_with_fusion()
        
        # 计算槽位偏移
        self.slot_offsets = []
        offset = 0
        for size in self.slot_sizes:
            self.slot_offsets.append(offset)
            offset += size
        
        # 检查是否超过限制
        if self.peak_memory > limit:
            print(f"  警告: 峰值内存 {self.peak_memory} bytes 超过限制 {limit} bytes")
        else:
            print(f"  ✓ 内存满足限制: {self.peak_memory} <= {limit} bytes")
        
        print(f"  乒乓缓冲区分析完成:")
        print(f"    - 内存限制: {limit} bytes ({limit/1024:.1f} KB)")
        print(f"    - 实际峰值内存: {self.peak_memory} bytes ({self.peak_memory/1024:.1f} KB)")
        print(f"    - 槽位数: {len(self.slot_sizes)}")
        if self.attention_blocks:
            print(f"    - 注意力块: {len(self.attention_blocks)} 个, 可融合: {len(fusable_blocks)} 个")
    
    def _compute_memory_reuse_allocation_with_fusion(self):
        """支持注意力融合的内存分配算法
        
        跳过融合注意力块的大型中间张量，它们在运行时逐头计算
        """
        # 按照首次使用时间排序
        sorted_tensors = sorted(
            self.activation_tensors,
            key=lambda tid: (self.tensor_lifetime.get(tid, (0, 0))[0], 
                           -self.tensors_info.get(tid, {}).get('size', 0))
        )
        
        # 槽位列表
        slots = []
        slot_assignments = {}
        
        # 零拷贝优化: 追踪哪些张量可以别名到其他张量
        # 格式: {output_tid: input_tid} 表示 output_tid 直接使用 input_tid 的内存
        self.zero_copy_aliases = {}
        
        # 更安全的零拷贝策略：让输出使用与输入相同的槽位
        # 这样 op_copy 会检测到 pin == pout 并跳过拷贝
        self.same_slot_pairs = {}  # {output_tid: input_tid}
        
        # 识别可以共享槽位的 reshape/expand_dims 操作
        for op in self.ops:
            op_type = op['type']
            if op_type in ('RESHAPE', 'EXPAND_DIMS'):
                input_tid = op['inputs'][0]
                output_tid = op['outputs'][0]
                
                # 检查输入和输出大小是否相同
                in_size = self.tensors_info.get(input_tid, {}).get('size', 0)
                out_size = self.tensors_info.get(output_tid, {}).get('size', 0)
                
                if in_size == out_size and in_size > 0:
                    # 检查输入张量是否在该操作后仍被使用
                    input_lifetime = self.tensor_lifetime.get(input_tid, (0, 0))
                    op_id = op['id']
                    
                    # 如果输入的最后使用就是当前操作，可以让输出共享槽位
                    if input_lifetime[1] == op_id:
                        self.same_slot_pairs[output_tid] = input_tid
        
        print(f"  零拷贝优化: 识别到 {len(self.same_slot_pairs)} 个可共享槽位的操作")
        
        # 第一遍：分配所有非共享槽位的张量
        for tid in sorted_tensors:
            if tid not in self.tensors_info:
                continue
            
            # 跳过融合注意力块的中间张量
            if tid in self.attention_tensors:
                continue
            
            # 如果该张量是零拷贝别名的目标，则跳过（稍后处理）
            if tid in self.zero_copy_aliases:
                continue
            
            # 跳过可共享槽位的张量（稍后处理）
            if tid in self.same_slot_pairs:
                continue
                
            size = self.tensors_info[tid]['size']
            start_time, end_time = self.tensor_lifetime.get(tid, (0, len(self.ops)))
            
            # 查找可复用的槽位
            best_slot = None
            best_slot_idx = -1
            best_waste = float('inf')
            
            for i, slot in enumerate(slots):
                if slot['end_time'] < start_time:
                    waste = abs(slot['size'] - size)
                    if waste < best_waste:
                        best_waste = waste
                        best_slot = slot
                        best_slot_idx = i
            
            if best_slot is not None:
                best_slot['size'] = max(best_slot['size'], size)
                best_slot['end_time'] = end_time
                slot_assignments[tid] = best_slot_idx
            else:
                slot_assignments[tid] = len(slots)
                slots.append({'size': size, 'end_time': end_time})
        
        # 第二遍：处理可共享槽位的张量
        # 按照依赖顺序处理，确保源张量已经被分配
        processed = set()
        changed = True
        while changed:
            changed = False
            for tid in sorted_tensors:
                if tid in processed or tid not in self.same_slot_pairs:
                    continue
                    
                source_tid = self.same_slot_pairs[tid]
                # 递归查找最终源
                while source_tid in self.same_slot_pairs:
                    source_tid = self.same_slot_pairs[source_tid]
                
                if source_tid in slot_assignments:
                    slot_assignments[tid] = slot_assignments[source_tid]
                    # 延长槽位的使用时间
                    slot_idx = slot_assignments[tid]
                    end_time = self.tensor_lifetime.get(tid, (0, 0))[1]
                    if slots[slot_idx]['end_time'] < end_time:
                        slots[slot_idx]['end_time'] = end_time
                    processed.add(tid)
                    changed = True
        
        # 第三遍：处理零拷贝别名（如果有）
        for tid in self.zero_copy_aliases:
            if tid in slot_assignments:
                continue
            source_tid = self.zero_copy_aliases[tid]
            while source_tid in self.zero_copy_aliases:
                source_tid = self.zero_copy_aliases[source_tid]
            
            if source_tid in slot_assignments:
                slot_assignments[tid] = slot_assignments[source_tid]
                slot_idx = slot_assignments[tid]
                end_time = self.tensor_lifetime.get(tid, (0, 0))[1]
                if slots[slot_idx]['end_time'] < end_time:
                    slots[slot_idx]['end_time'] = end_time
        
        slot_sizes = [s['size'] for s in slots]
        peak_memory = sum(slot_sizes)
        
        # 为逐头注意力计算添加临时缓冲区
        # 需要存储: 单头注意力矩阵 (187*187) + softmax 行缓冲 (187*4)
        if self.attention_blocks:
            sample_block = self.attention_blocks[0]
            per_head_temp = sample_block['seq_len'] * sample_block['seq_len']  # 35KB
            per_head_temp += sample_block['seq_len'] * 4  # softmax float buffer
            slot_sizes.append(per_head_temp)
            peak_memory += per_head_temp
        
        return slot_assignments, slot_sizes, peak_memory
    
    def _identify_attention_blocks(self):
        """识别可融合的注意力操作块
        
        分析注意力块结构并确定是否可以通过操作重排实现融合
        """
        self.attention_blocks = []
        self.attention_tensors = set()  # 大型注意力中间张量（可跳过分配）
        self.fused_ops = set()  # 被融合的操作ID
        self.reordered_ops = []  # 重排后的操作列表
        
        i = 0
        while i < len(self.ops):
            op = self.ops[i]
            
            if op['type'] == 'BATCH_MATMUL':
                out_tid = op['outputs'][0]
                out_info = self.tensors_info.get(out_tid, {})
                out_shape = out_info.get('shape', ())
                
                # 检查是否是 QK^T 矩阵乘法 (输出是注意力分数)
                if len(out_shape) == 4 and out_shape[2] == out_shape[3]:
                    seq_len = int(out_shape[2])
                    num_heads = int(out_shape[1])
                    
                    # 尝试匹配完整的注意力块
                    block = self._try_match_attention_block_with_reorder(i, num_heads, seq_len)
                    if block:
                        self.attention_blocks.append(block)
                        for tid in block['intermediate_tensors']:
                            self.attention_tensors.add(tid)
                        for op_id in block['fused_op_ids']:
                            self.fused_ops.add(op_id)
                        # 记录需要提前执行的 V 计算操作
                        block['v_prep_ops'] = block.get('v_prep_ops', [])
                        i = block['end_op'] + 1
                        continue
            i += 1
    
    def _try_match_attention_block_with_reorder(self, start_idx, num_heads, seq_len):
        """尝试匹配注意力块并分析是否可以重排 V 的计算
        
        期望的操作序列:
        1. BATCH_MATMUL (Q @ K^T) -> (1, 8, 187, 187)
        2. TRANSPOSE -> (1, 8, 187, 187)
        3. SOFTMAX -> (1, 8, 187, 187)
        ... V的计算操作 ...
        4. BATCH_MATMUL (Attn @ V) -> (1, 8, 187, 16)
        """
        if start_idx + 3 >= len(self.ops):
            return None
        
        op_qk = self.ops[start_idx]
        qk_out_tid = op_qk['outputs'][0]
        qk_size = self.tensors_info.get(qk_out_tid, {}).get('size', 0)
        
        # 找 TRANSPOSE
        op_trans = self.ops[start_idx + 1]
        if op_trans['type'] != 'TRANSPOSE':
            return None
        trans_out_tid = op_trans['outputs'][0]
        
        # 找 SOFTMAX
        op_softmax = self.ops[start_idx + 2]
        if op_softmax['type'] != 'SOFTMAX':
            return None
        softmax_out_tid = op_softmax['outputs'][0]
        
        # 找使用 softmax 输出的 BATCH_MATMUL (Attn @ V)
        attn_v_idx = None
        v_prep_ops = []  # V 的准备操作
        
        for j in range(start_idx + 3, min(start_idx + 10, len(self.ops))):
            op = self.ops[j]
            if op['type'] == 'BATCH_MATMUL':
                if softmax_out_tid in op['inputs']:
                    attn_v_idx = j
                    break
            else:
                # 这些是 V 的准备操作
                v_prep_ops.append(j)
        
        if attn_v_idx is None:
            return None
        
        op_av = self.ops[attn_v_idx]
        av_out_tid = op_av['outputs'][0]
        
        # 获取 Q, K, V 张量
        q_tid, k_tid = op_qk['inputs'][0], op_qk['inputs'][1]
        v_candidates = [t for t in op_av['inputs'] if t != softmax_out_tid]
        v_tid = v_candidates[0] if v_candidates else None
        
        if v_tid is None:
            return None
        
        # 检查 V 的准备操作是否可以提前执行
        # V 准备操作的**外部**输入依赖必须在 QK 之前就已经可用
        # 内部依赖（V 准备操作之间的依赖）是可以的
        can_reorder = True
        v_prep_outputs = set()  # V 准备操作产生的张量
        for op_idx in v_prep_ops:
            op = self.ops[op_idx]
            for out_tid in op['outputs']:
                v_prep_outputs.add(out_tid)
        
        for op_idx in v_prep_ops:
            op = self.ops[op_idx]
            for in_tid in op['inputs']:
                # 跳过 V 准备操作内部产生的张量
                if in_tid in v_prep_outputs:
                    continue
                if in_tid in self.tensor_lifetime:
                    first_use, last_use = self.tensor_lifetime[in_tid]
                    # 如果**外部**输入在 QK 之后才产生，无法重排
                    if first_use >= start_idx:
                        can_reorder = False
                        break
            if not can_reorder:
                break
        
        return {
            'start_op': start_idx,
            'end_op': attn_v_idx,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'q_tid': q_tid,
            'k_tid': k_tid,
            'v_tid': v_tid,
            'output_tid': av_out_tid,
            'intermediate_tensors': [qk_out_tid, trans_out_tid, softmax_out_tid],
            'fused_op_ids': [start_idx, start_idx + 1, start_idx + 2, attn_v_idx],
            'v_prep_ops': v_prep_ops,
            'can_reorder': can_reorder,
            'qk_size': qk_size
        }

    def _precompute_q16_scales(self):
        """预扫描所有操作，收集 per-channel Q16 scales (适配26位中间结果)"""
        self.q16_scales = {}  # {op_index: {'weight_tid': tid, 'scales_q16': [...]}}
        
        for op in self.ops:
            op_type = op['type']
            op_idx = op['id']
            inputs = op['inputs']
            outputs = op['outputs']
            
            if op_type in ('FULLY_CONNECTED', 'CONV_2D'):
                if len(inputs) < 2:
                    continue
                    
                weight_tid = inputs[1]
                out_tid = outputs[0] if len(outputs) > 0 else -1
                
                # 获取输入和输出的 scale
                si, zi = self._get_scale_zp(inputs[0])
                so, zo = self._get_scale_zp(out_tid)
                
                # 获取权重的 per-channel scales
                weight_info = self.tensors_info.get(weight_tid, {})
                weight_scales = weight_info.get('scales', np.array([]))
                
                if len(weight_scales) > 1:
                    # 预计算 Q16: scale_q16[o] = (si * weight_scale[o] / so) * (1<<16)
                    # Q16 适配 INT26 中间结果: acc(~20bit) * scale(~16bit) >> 16 = ~20bit
                    scales_q16 = []
                    for ws in weight_scales:
                        eff = (si * ws / so) * (1 << 16)
                        scales_q16.append(int(round(eff)))
                    
                    self.q16_scales[op_idx] = {
                        'weight_tid': weight_tid,
                        'scales_q16': scales_q16,
                        'chn_out': len(weight_scales)
                    }

    def _get_scale_zp(self, tid):
        """获取张量的scale和zero_point"""
        info = self.tensors_info.get(tid, {})
        scales = info.get('scales', np.array([]))
        zps = info.get('zps', np.array([]))
        if len(scales) > 0:
            return float(scales[0]), int(zps[0]) if len(zps) > 0 else 0
        return 1.0, 0
    
    def generate(self, output_dir: str, memory_limit: int = None):
        """生成模块化的C实现（内存优化版，峰值约152KB）
        
        Args:
            output_dir: 输出目录
            memory_limit: 内存限制（字节），默认262144 (256KB)
        
        生成的文件:
        - ecgformer_config.h    : 模型配置和类型定义
        - ecgformer_weights.h   : INT8权重数据
        - ecgformer_biases.h    : INT32偏置数据
        - ecgformer_quant.h     : 量化参数（scale和zero_point）
        - ecgformer_ops.h       : 操作函数实现
        - ecgformer_model.c     : 主程序（推理函数和main）
        """
        self.memory_mode = 'pingpong'  # 固定使用 pingpong 模式
        self.memory_limit = memory_limit or 262144  # 默认256KB
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算内存分配（使用注意力融合优化）
        self._compute_pingpong_allocation()
        
        # 1. 生成配置头文件
        self._generate_config_header(output_dir)
        
        # 2. 生成权重头文件
        self._generate_weights_header(output_dir)
        
        # 3. 生成偏置头文件
        self._generate_biases_header(output_dir)
        
        # 4. 生成量化参数头文件
        self._generate_quant_header(output_dir)
        
        # 5. 生成操作头文件
        self._generate_ops_header(output_dir)
        
        # 6. 生成主程序
        self._generate_main_source(output_dir)
        
        print(f"\n代码生成完成: {output_dir}")
        print(f"  操作数: {len(self.ops)}")
        print(f"  常量张量: {len(self.constant_tensors)}")
        print(f"  激活张量: {len(self.activation_tensors)}")
        print(f"  峰值内存: {self.peak_memory} bytes ({self.peak_memory/1024:.1f} KB)")
        print(f"  内存限制: {self.memory_limit} bytes ({self.memory_limit/1024:.1f} KB)")
        print(f"\n生成的文件:")
        print(f"  - ecgformer_config.h  : 模型配置")
        print(f"  - ecgformer_weights.h : 权重数据")
        print(f"  - ecgformer_biases.h  : 偏置数据")
        print(f"  - ecgformer_quant.h   : 量化参数")
        print(f"  - ecgformer_ops.h     : 操作函数")
        print(f"  - ecgformer_model.c   : 主程序")
        print(f"\n编译命令:")
        print(f"  gcc -O3 -o ecgformer ecgformer_model.c -lm")
    
    def _generate_config_header(self, output_dir: str):
        """生成配置头文件"""
        input_shape = tuple(self.input_details[0]['shape'])
        output_shape = tuple(self.output_details[0]['shape'])
        input_scale, input_zp = self._get_scale_zp(self.input_details[0]['index'])
        output_scale, output_zp = self._get_scale_zp(self.output_details[0]['index'])
        
        # 根据内存模式选择池大小
        memory_mode = getattr(self, 'memory_mode', 'static')
        if memory_mode == 'pingpong':
            pool_size = self.memory_limit
            buffer_size = pool_size // 2
            num_slots = 0
        elif memory_mode == 'reuse':
            pool_size = self.peak_memory
            num_slots = len(self.slot_sizes)
            buffer_size = 0
        else:
            pool_size = sum(self.tensors_info[tid]['size'] 
                           for tid in self.activation_tensors 
                           if tid in self.tensors_info)
            num_slots = 0
            buffer_size = 0
        
        # 计算注意力相关参数
        num_heads = 8  # 默认值
        seq_len = 187  # 默认值
        head_dim = 16  # 默认值
        if hasattr(self, 'attention_blocks') and self.attention_blocks:
            sample_block = self.attention_blocks[0]
            num_heads = sample_block['num_heads']
            seq_len = sample_block['seq_len']
            head_dim = 128 // num_heads  # 假设总维度是128
        elif hasattr(self, 'per_head_sizes') and self.per_head_sizes:
            sample_tid = list(self.per_head_sizes.keys())[0]
            sample_info = self.per_head_sizes[sample_tid]
            num_heads = sample_info['num_heads']
            seq_len = sample_info['seq_len']
            head_dim = 128 // num_heads
        
        code = f'''/**
 * ECGformer INT8 模型配置
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_CONFIG_H
#define ECGFORMER_CONFIG_H

#include <stdint.h>

// ============== 模型配置 ==============

#define INPUT_SIZE {int(np.prod(input_shape))}
#define OUTPUT_CLASSES {output_shape[-1]}
#define NUM_TENSORS {len(self.tensors_info)}

// 内存管理模式
#define MEMORY_MODE_STATIC 0
#define MEMORY_MODE_REUSE 1
#define MEMORY_MODE_PINGPONG 2
#define CURRENT_MEMORY_MODE {0 if memory_mode == "static" else (1 if memory_mode == "reuse" else 2)}
'''
        
        if memory_mode == 'pingpong':
            # pingpong 模式使用 slot 分配，需要 slot 偏移数组
            num_slots = len(self.slot_sizes) if hasattr(self, 'slot_sizes') else 0
            code += f'''
// ============== 乒乓缓冲区模式 (严格256KB内存限制) ==============
// 策略: 槽位复用 + 融合注意力层 (逐Head计算)
// 融合后避免存储 (8, 187, 187) = 280KB 的注意力矩阵

#define MEMORY_LIMIT {pool_size}
#define NUM_MEMORY_SLOTS {num_slots}

// 注意力计算参数
#define NUM_HEADS {num_heads}
#define SEQ_LEN {seq_len}
#define HEAD_DIM {head_dim}
#define ATTENTION_PER_HEAD (SEQ_LEN * SEQ_LEN)  // {seq_len * seq_len} bytes per head

// 总内存池 (用于激活张量)
#define ACTIVATION_POOL_SIZE {self.peak_memory}
'''
            # 添加 slot 偏移数组
            if hasattr(self, 'slot_offsets') and self.slot_offsets:
                code += f'''
// 槽位大小数组
static const int g_slot_sizes[{num_slots}] = {{
    {', '.join(str(s) for s in self.slot_sizes)}
}};

// 槽位偏移数组 (预计算)
static const int g_slot_offsets[{num_slots}] = {{
    {', '.join(str(o) for o in self.slot_offsets)}
}};
'''
        elif memory_mode == 'reuse':
            code += f'''
// 槽位复用模式: 每个槽位可被多个张量复用
#define NUM_MEMORY_SLOTS {num_slots}
#define ACTIVATION_POOL_SIZE {pool_size}

// 槽位大小数组
static const int g_slot_sizes[{num_slots}] = {{
    {', '.join(str(s) for s in self.slot_sizes)}
}};

// 槽位偏移数组 (预计算)
'''
            # 计算槽位偏移
            slot_offsets = []
            offset = 0
            for s in self.slot_sizes:
                slot_offsets.append(offset)
                offset += s
            code += f'''static const int g_slot_offsets[{num_slots}] = {{
    {', '.join(str(o) for o in slot_offsets)}
}};
'''
        else:
            code += f'''#define ACTIVATION_POOL_SIZE {pool_size}
'''
        
        code += f'''
// 输入量化参数
#define INPUT_SCALE {input_scale:.10e}f
#define INPUT_ZERO_POINT {input_zp}

// 输出量化参数
#define OUTPUT_SCALE {output_scale:.10e}f
#define OUTPUT_ZERO_POINT {output_zp}

// 类别名称
static const char* CLASS_NAMES[5] = {{"N (正常)", "S (室上性)", "V (室性)", "F (融合)", "Q (未知)"}};

#endif // ECGFORMER_CONFIG_H
'''
        
        path = os.path.join(output_dir, 'ecgformer_config.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_config.h")
    
    def _generate_weights_header(self, output_dir: str):
        """生成权重头文件（INT8权重）"""
        code = '''/**
 * ECGformer INT8 权重数据
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_WEIGHTS_H
#define ECGFORMER_WEIGHTS_H

#include <stdint.h>

// ============== INT8 权重数据 ==============

'''
        
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            
            data = self.weights_data[tid]
            info = self.tensors_info[tid]
            dtype = info['dtype']
            size = info['size']
            
            # 只处理INT8权重
            if 'int8' in dtype:
                flat = data.flatten().astype(np.int8)
                shape_str = 'x'.join(str(s) for s in info['shape'])
                code += f'// 张量 {tid}: 形状 [{shape_str}]\n'
                code += f'static const int8_t weight_t{tid}[{size}] = {{\n    '
                for i, v in enumerate(flat):
                    code += f'{v}'
                    if i < len(flat) - 1:
                        code += ', '
                        if (i + 1) % 20 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        code += '#endif // ECGFORMER_WEIGHTS_H\n'
        
        path = os.path.join(output_dir, 'ecgformer_weights.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_weights.h")
    
    def _generate_biases_header(self, output_dir: str):
        """生成偏置头文件（INT32偏置）"""
        code = '''/**
 * ECGformer INT32 偏置数据
 * 自动生成 - 请勿手动修改
 */

#ifndef ECGFORMER_BIASES_H
#define ECGFORMER_BIASES_H

#include <stdint.h>

// ============== INT32 偏置数据 ==============

'''
        
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            
            data = self.weights_data[tid]
            info = self.tensors_info[tid]
            dtype = info['dtype']
            size = info['size']
            
            # 只处理INT32偏置
            if 'int32' in dtype:
                flat = data.flatten().astype(np.int32)
                shape_str = 'x'.join(str(s) for s in info['shape'])
                code += f'// 张量 {tid}: 形状 [{shape_str}]\n'
                code += f'static const int32_t bias_t{tid}[{size}] = {{\n    '
                for i, v in enumerate(flat):
                    code += f'{v}'
                    if i < len(flat) - 1:
                        code += ', '
                        if (i + 1) % 10 == 0:
                            code += '\n    '
                code += '\n};\n\n'
        
        code += '#endif // ECGFORMER_BIASES_H\n'
        
        path = os.path.join(output_dir, 'ecgformer_biases.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_biases.h")
    
    def _generate_quant_header(self, output_dir: str):
        """生成量化参数头文件 - Bare-metal Style (无结构体, Q16格式适配INT26)"""
        code = '''/**
 * ECGformer 量化参数 - Bare-metal Style
 * 自动生成 - 请勿手动修改
 * 
 * 注意: 不使用结构体, 使用扁平数组以便直接硬件访问
 * Q16格式: scale = (int32_t)((si * weight_scale / so) * (1<<16))
 * 适配INT26中间结果: acc(~20bit) * scale(~16bit) = ~36bit, >> 16后 = ~20bit
 */

#ifndef ECGFORMER_QUANT_H
#define ECGFORMER_QUANT_H

#include <stdint.h>

// ============== 预计算的 Per-Channel Q16 Scales ==============
// Q16格式: 已预计算 (si * weight_scale[o] / so) * (1<<16)
// 每个操作有独立的 Q16 scale 数组 (以操作索引命名)
// 使用 int32_t 适配嵌入式硬件

'''
        
        # 生成预计算的 Q16 scales (以操作索引命名)
        for op_idx in sorted(self.q16_scales.keys()):
            info = self.q16_scales[op_idx]
            scales_q16 = info['scales_q16']
            chn_out = info['chn_out']
            weight_tid = info['weight_tid']
            
            code += f'// Op#{op_idx} 的 per-channel Q16 scales (权重张量 {weight_tid}, {chn_out} channels)\n'
            code += f'static const int32_t pscales_q16_op{op_idx}[{chn_out}] = {{\n    '
            for i, sq16 in enumerate(scales_q16):
                code += f'{sq16}'
                if i < chn_out - 1:
                    code += ', '
                    if (i + 1) % 4 == 0:
                        code += '\n    '
            code += '\n};\n\n'
        
        code += '''
#endif // ECGFORMER_QUANT_H
'''
        
        path = os.path.join(output_dir, 'ecgformer_quant.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_quant.h")
    
    def _generate_ops_header(self, output_dir: str):
        """生成操作函数头文件 - Bare-metal Hardware Verification Style"""
        code = '''/**
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

// ============== 乒乓缓冲区模式专用函数 ==============

#if CURRENT_MEMORY_MODE == MEMORY_MODE_PINGPONG

// 逐Head批量矩阵乘法 (用于256KB内存限制模式)
// 对于大型注意力矩阵，一次只处理一个head，避免内存溢出
// 输入维度: pin1 [num_heads, seq_len, head_dim], pin2 [num_heads, head_dim, seq_len] (或K^T)
// 输出维度: pout [num_heads, seq_len, seq_len] (但每次只计算一个head到临时缓冲)
// ptemp: 临时缓冲区，大小至少 seq_len * seq_len
static void op_batch_matmul_per_head(
    const int8_t* pin1, const int8_t* pin2, int8_t* pout, int8_t* ptemp,
    int num_heads, int seq_len, int head_dim, int side_n,
    int32_t scale, int z1, int z2, int zo) {
    
    HW_BARRIER();
    
    int stride1_h = seq_len * head_dim;  // Q/K 每个head的stride
    int stride2_h = head_dim * side_n;   // K^T/V 每个head的stride (side_n可能是seq_len或head_dim)
    int stride_out = seq_len * side_n;   // 输出每个head的stride
    
    // 逐Head处理，避免同时持有所有head的结果
    for (volatile int h = 0; h < num_heads; h++) {
        int base1 = h * stride1_h;
        int base2 = h * stride2_h;
        int base_out = h * stride_out;
        
        // 计算当前head的矩阵乘法结果到临时缓冲区
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < side_n; j++) {
                int32_t t0 = 0;  // 累加器
                for (volatile int l = 0; l < head_dim; l++) {
                    int32_t t1 = (int32_t)*(pin1 + base1 + i * head_dim + l) - z1;
                    int32_t t2 = (int32_t)*(pin2 + base2 + l * side_n + j) - z2;
                    t0 += t1 * t2;
                }
                int32_t t3 = ((t0 >> 8) * scale) >> 8;
                // 直接写入最终输出位置
                *(pout + base_out + i * side_n + j) = saturate_int8(t3 + zo);
            }
        }
        HW_BARRIER();  // 确保每个head计算完成
    }
}

// 逐Head注意力计算 (融合 Q*K^T + Softmax + *V)
// 这是内存高效的注意力实现，一次只处理一个head
// pQ: [num_heads, seq_len, head_dim]
// pK: [num_heads, seq_len, head_dim] (会被转置为 [num_heads, head_dim, seq_len])
// pV: [num_heads, seq_len, head_dim]
// pout: [num_heads, seq_len, head_dim]
// ptemp: 临时缓冲区，大小至少 seq_len * seq_len (用于单个head的注意力矩阵)
static void op_attention_per_head(
    const int8_t* pQ, const int8_t* pK, const int8_t* pV, int8_t* pout, int8_t* ptemp,
    int num_heads, int seq_len, int head_dim,
    int32_t scale_qk, int zq, int zk,      // Q*K^T 的量化参数
    float scale_softmax_in, int z_softmax,  // Softmax输入的量化参数
    int32_t scale_av, int za, int zv,       // Attn*V 的量化参数  
    float scale_softmax_out,                // Softmax输出的量化参数
    int zo) {
    
    HW_BARRIER();
    
    int stride_qkv = seq_len * head_dim;
    int stride_out = seq_len * head_dim;
    int attn_size = seq_len * seq_len;
    
    // 分配临时浮点缓冲区用于softmax
    float* pvals = (float*)malloc(seq_len * sizeof(float));
    
    // 逐Head处理
    for (volatile int h = 0; h < num_heads; h++) {
        int base_q = h * stride_qkv;
        int base_k = h * stride_qkv;
        int base_v = h * stride_qkv;
        int base_out = h * stride_out;
        
        // Step 1: 计算 Q * K^T -> ptemp [seq_len, seq_len]
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < seq_len; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < head_dim; l++) {
                    int32_t q_val = (int32_t)*(pQ + base_q + i * head_dim + l) - zq;
                    int32_t k_val = (int32_t)*(pK + base_k + j * head_dim + l) - zk;  // 注意K是按行读取，等效于K^T的列
                    t0 += q_val * k_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_qk) >> 8;
                *(ptemp + i * seq_len + j) = saturate_int8(t3);
            }
        }
        
        // Step 2: 对每一行应用Softmax (原地，使用浮点)
        for (volatile int i = 0; i < seq_len; i++) {
            int row_base = i * seq_len;
            
            // 找最大值
            float max_val = -1e9f;
            for (volatile int j = 0; j < seq_len; j++) {
                float val = ((float)(*(ptemp + row_base + j)) - (float)z_softmax) * scale_softmax_in;
                *(pvals + j) = val;  // 临时存储反量化值
                if (val > max_val) max_val = val;
            }
            
            // 计算exp并求和
            float sum = 0.0f;
            for (volatile int j = 0; j < seq_len; j++) {
                float exp_val = expf(*(pvals + j) - max_val);
                *(pvals + j) = exp_val;
                sum += exp_val;
            }
            
            // 归一化并写回
            if (sum == 0.0f) sum = 1e-10f;
            for (volatile int j = 0; j < seq_len; j++) {
                float prob = *(pvals + j) / sum;
                int32_t q_val = (int32_t)roundf(prob / scale_softmax_out) + z_softmax;
                *(ptemp + row_base + j) = saturate_int8(q_val);
            }
        }
        
        // Step 3: 计算 Attn * V -> pout [seq_len, head_dim]
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < head_dim; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < seq_len; l++) {
                    int32_t a_val = (int32_t)*(ptemp + i * seq_len + l) - za;
                    int32_t v_val = (int32_t)*(pV + base_v + l * head_dim + j) - zv;
                    t0 += a_val * v_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_av) >> 8;
                *(pout + base_out + i * head_dim + j) = saturate_int8(t3 + zo);
            }
        }
        
        HW_BARRIER();
    }
    
    free(pvals);
}

// 融合注意力层: Q @ K^T -> Transpose -> Softmax -> @ V
// 完全逐Head计算，只需要 seq_len * seq_len bytes 临时缓冲
// pQ: [1, num_heads, seq_len, head_dim]
// pK: [1, num_heads, head_dim, seq_len] (已转置)
// pV: [1, num_heads, seq_len, head_dim]
// pout: [1, num_heads, seq_len, head_dim]
static void op_fused_attention_per_head(
    const int8_t* pQ, const int8_t* pK, const int8_t* pV, int8_t* pout,
    int num_heads, int seq_len, int head_dim,
    int32_t scale_qk, int zq, int zk, int z_qk_out,  // Q*K^T 量化参数
    float scale_qk_out, float scale_softmax_out, int z_softmax,  // Softmax 量化参数
    int32_t scale_av, int zv, int zo) {  // Attn*V 量化参数
    
    HW_BARRIER();
    
    // 计算步长
    int stride_q = seq_len * head_dim;  // Q: (num_heads, seq_len, head_dim)
    int stride_k = head_dim * seq_len;  // K: (num_heads, head_dim, seq_len) - 已转置
    int stride_v = seq_len * head_dim;  // V: (num_heads, seq_len, head_dim)
    int stride_out = seq_len * head_dim;
    
    // 静态临时缓冲区 (避免malloc)
    // 使用全局静态缓冲区，大小足够存储一个head的注意力矩阵
    static int8_t s_attn_temp[187 * 187];  // seq_len * seq_len
    static float s_row_temp[187];  // seq_len，用于softmax行计算
    
    // 逐Head处理
    for (volatile int h = 0; h < num_heads; h++) {
        int base_q = h * stride_q;
        int base_k = h * stride_k;
        int base_v = h * stride_v;
        int base_out = h * stride_out;
        
        // Step 1: 计算 Q @ K^T -> attn_scores [seq_len, seq_len]
        // K的布局是 (num_heads, head_dim, seq_len)，相当于已经是K^T
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < seq_len; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < head_dim; l++) {
                    // Q[h, i, l] @ K[h, l, j]
                    int32_t q_val = (int32_t)*(pQ + base_q + i * head_dim + l) - zq;
                    int32_t k_val = (int32_t)*(pK + base_k + l * seq_len + j) - zk;
                    t0 += q_val * k_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_qk) >> 8;
                // 注意力分数存储在临时缓冲区，使用转置后的索引 (j, i)
                // 因为 TFLite 中的 TRANSPOSE 在 BATCH_MATMUL 之后
                s_attn_temp[j * seq_len + i] = saturate_int8(t3 + z_qk_out);
            }
        }
        
        // Step 2: 对每一行应用 Softmax (转置后的注意力矩阵)
        for (volatile int i = 0; i < seq_len; i++) {
            int row_base = i * seq_len;
            
            // 反量化并找最大值
            float max_val = -1e9f;
            for (volatile int j = 0; j < seq_len; j++) {
                float val = ((float)(s_attn_temp[row_base + j]) - (float)z_qk_out) * scale_qk_out;
                s_row_temp[j] = val;
                if (val > max_val) max_val = val;
            }
            
            // 计算 exp 并求和
            float sum = 0.0f;
            for (volatile int j = 0; j < seq_len; j++) {
                float exp_val = expf(s_row_temp[j] - max_val);
                s_row_temp[j] = exp_val;
                sum += exp_val;
            }
            
            // 归一化并量化写回
            if (sum < 1e-10f) sum = 1e-10f;
            for (volatile int j = 0; j < seq_len; j++) {
                float prob = s_row_temp[j] / sum;
                int32_t q_val = (int32_t)roundf(prob / scale_softmax_out) + z_softmax;
                s_attn_temp[row_base + j] = saturate_int8(q_val);
            }
        }
        
        // Step 3: 计算 Attn @ V -> output [seq_len, head_dim]
        for (volatile int i = 0; i < seq_len; i++) {
            for (volatile int j = 0; j < head_dim; j++) {
                int32_t t0 = 0;
                for (volatile int l = 0; l < seq_len; l++) {
                    // attn[i, l] @ V[h, l, j]
                    int32_t a_val = (int32_t)s_attn_temp[i * seq_len + l] - z_softmax;
                    int32_t v_val = (int32_t)*(pV + base_v + l * head_dim + j) - zv;
                    t0 += a_val * v_val;
                }
                int32_t t3 = ((t0 >> 8) * scale_av) >> 8;
                *(pout + base_out + i * head_dim + j) = saturate_int8(t3 + zo);
            }
        }
        
        HW_BARRIER();
    }
}

#endif // CURRENT_MEMORY_MODE == MEMORY_MODE_PINGPONG

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
'''
        
        path = os.path.join(output_dir, 'ecgformer_ops.h')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_ops.h")
    
    def _generate_main_source(self, output_dir: str):
        """生成主程序源文件 - Bare-metal Hardware Verification Style"""
        input_tid = self.input_details[0]['index']
        output_tid = self.output_details[0]['index']
        memory_mode = getattr(self, 'memory_mode', 'static')
        
        # 计算每个激活张量的偏移
        if memory_mode == 'pingpong':
            # 乒乓缓冲模式: 使用双缓冲区
            buffer_size = self.memory_limit // 2
            offsets = {}
            # 简化分配: 所有激活从缓冲区A开始，动态切换
            for tid in sorted(self.activation_tensors):
                if tid in self.tensors_info:
                    # 先全部放在缓冲区A（偏移0），实际运行时动态决定
                    offsets[tid] = 0
        elif memory_mode == 'reuse':
            # 槽位复用模式: 使用槽位偏移
            slot_offsets = []
            offset = 0
            for s in self.slot_sizes:
                slot_offsets.append(offset)
                offset += s
            offsets = {}
            for tid in self.activation_tensors:
                if tid in self.slot_assignments:
                    slot_idx = self.slot_assignments[tid]
                    offsets[tid] = slot_offsets[slot_idx]
        else:
            # 静态模式: 顺序分配
            offsets = {}
            offset = 0
            for tid in sorted(self.activation_tensors):
                if tid in self.tensors_info:
                    offsets[tid] = offset
                    offset += self.tensors_info[tid]['size']
        
        # 确定内存模式描述
        if memory_mode == 'pingpong':
            mode_desc = f"pingpong (乒乓双缓冲, 限制{self.memory_limit//1024}KB)"
        elif memory_mode == 'reuse':
            mode_desc = "reuse (槽位复用)"
        else:
            mode_desc = "static (静态分配)"
        
        code = f'''/**
 * ECGformer INT8 主程序 - Bare-metal Hardware Verification Style
 * 自动生成 - 请勿手动修改
 * 
 * 内存模式: {mode_desc}
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

// ============== 张量存储 ==============

// 激活张量内存池
static int8_t g_activation_pool[ACTIVATION_POOL_SIZE];

// 张量指针数组 (扁平int8_t*)
static int8_t* ptensors[NUM_TENSORS];
'''
        
        # pingpong模式现在使用slot分配（与reuse相同），不需要额外的缓冲区管理变量
        
        code += '''
// 初始化张量指针 (无null检查, 直接赋值)
static void init_tensors(void) {
'''
        
        # 常量张量指向静态数组 (使用指针算术)
        for tid in sorted(self.constant_tensors):
            if tid not in self.weights_data:
                continue
            info = self.tensors_info[tid]
            dtype = info['dtype']
            if 'int8' in dtype:
                code += f'    *(ptensors + {tid}) = (int8_t*)weight_t{tid};\n'
            elif 'int32' in dtype:
                code += f'    *(ptensors + {tid}) = (int8_t*)bias_t{tid};\n'
        
        if memory_mode == 'pingpong':
            # pingpong模式使用slot分配
            input_slot_idx = self.slot_assignments.get(input_tid, 0)
            code += f'''    // 融合注意力模式 + 槽位复用: 输入张量预先设置
    *(ptensors + {input_tid}) = g_activation_pool + g_slot_offsets[{input_slot_idx}];
    HW_BARRIER();  // 内存屏障确保初始化完成
}}

// ============== 推理函数 ==============

int ecgformer_inference(const float* pinput_float, float* poutput_probs) {{
'''
        elif memory_mode == 'reuse':
            # 槽位复用模式: 在推理开始时动态设置指针
            input_slot_idx = self.slot_assignments.get(input_tid, 0)
            code += f'''    // 槽位复用模式: 输入张量预先设置
    *(ptensors + {input_tid}) = g_activation_pool + g_slot_offsets[{input_slot_idx}];
    HW_BARRIER();  // 内存屏障确保初始化完成
}}

// ============== 推理函数 ==============

int ecgformer_inference(const float* pinput_float, float* poutput_probs) {{
'''
        else:
            # 静态模式: 预先分配
            for tid in sorted(self.activation_tensors):
                if tid in offsets:
                    code += f'    *(ptensors + {tid}) = g_activation_pool + {offsets[tid]};\n'
            
            code += '''    HW_BARRIER();  // 内存屏障确保初始化完成
}

// ============== 推理函数 ==============

int ecgformer_inference(const float* pinput_float, float* poutput_probs) {
'''
        
        code += f'''    int8_t* pin = *(ptensors + {input_tid});
    
    // 量化输入: 使用指针算术和volatile循环
    for (volatile int i = 0; i < INPUT_SIZE; i++) {{
        float t0 = *(pinput_float + i);
        *(pin + i) = quantize_float(t0, INPUT_SCALE, INPUT_ZERO_POINT);
    }}
    HW_BARRIER();
    
'''
        
        # 生成每个操作（支持注意力融合和操作重排）
        processed_ops = set()  # 已处理的操作ID
        
        for op in self.ops:
            op_id = op['id']
            
            # 跳过已经作为 V 准备操作处理过的操作
            if op_id in processed_ops:
                continue
            
            # 检查是否是可融合注意力块的起始操作
            is_attention_start = False
            for block in getattr(self, 'attention_blocks', []):
                if block['start_op'] == op_id and block.get('can_reorder', False):
                    is_attention_start = True
                    
                    # 先执行 V 的准备操作（重排）
                    v_prep_ops = block.get('v_prep_ops', [])
                    if v_prep_ops:
                        code += f'    // ===== V 准备操作 (提前执行) =====\n'
                        for v_op_id in v_prep_ops:
                            v_op = self.ops[v_op_id]
                            code += self._generate_op_code_modular(v_op)
                            processed_ops.add(v_op_id)
                    
                    # 生成融合注意力计算代码
                    code += self._generate_fused_attention_code(block)
                    
                    # 标记融合的操作为已处理
                    for fused_id in block['fused_op_ids']:
                        processed_ops.add(fused_id)
                    break
            
            if not is_attention_start:
                # 检查是否是被融合的操作（非起始）
                is_fused = op_id in getattr(self, 'fused_ops', set())
                if is_fused:
                    code += f'    // Op#{op_id}: {op["type"]} (fused in attention block)\n'
                else:
                    code += self._generate_op_code_modular(op)
        
        # 输出处理
        code += f'''
    // 反量化输出并找预测类别
    int8_t* pout = *(ptensors + {output_tid});
    int t0 = 0;  // pred
    float t1 = -1e9f;  // max_prob
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {{
        float t2 = dequantize_int8(*(pout + i), OUTPUT_SCALE, OUTPUT_ZERO_POINT);
        *(poutput_probs + i) = t2;
        if (t2 > t1) {{
            t1 = t2;
            t0 = i;
        }}
    }}
    return t0;
}}

// 获取INT8输出 (用于硬件验证)
void ecgformer_get_int8_output(int8_t* poutput) {{
    int8_t* psrc = *(ptensors + {output_tid});
    // 使用memcpy替代循环
    memcpy(poutput, psrc, OUTPUT_CLASSES);
}}

// ============== 共享库接口 ==============

#ifdef BUILD_SHARED_LIB
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void c_init(void) {{
    init_tensors();
}}

EXPORT int c_inference(const float* pinput, float* poutput) {{
    return ecgformer_inference(pinput, poutput);
}}

EXPORT void c_get_int8_output(int8_t* poutput) {{
    ecgformer_get_int8_output(poutput);
}}
#endif

// ============== 主函数 ==============

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {{
    init_tensors();
    
    printf("ECGformer INT8 Bare-metal C Implementation\\n");
    printf("==========================================\\n");
    
    // 测试用随机输入 (使用malloc, 无null检查)
    float* ptest_input = (float*)malloc(INPUT_SIZE << 2);  // INPUT_SIZE * 4
    for (volatile int i = 0; i < INPUT_SIZE; i++) {{
        *(ptest_input + i) = ((float)rand() / RAND_MAX - 0.5f);
    }}
    
    // 推理
    float* poutput_probs = (float*)malloc(OUTPUT_CLASSES << 2);  // OUTPUT_CLASSES * 4
    int t0 = ecgformer_inference(ptest_input, poutput_probs);
    
    printf("\\nPrediction Results:\\n");
    for (volatile int i = 0; i < OUTPUT_CLASSES; i++) {{
        printf("  Class %d (%s): %.4f%s\\n", i, CLASS_NAMES[i], *(poutput_probs + i),
               i == t0 ? " <-- Predicted" : "");
    }}
    
    free(ptest_input);
    free(poutput_probs);
    return 0;
}}
#endif
'''
        
        path = os.path.join(output_dir, 'ecgformer_model.c')
        with open(path, 'w') as f:
            f.write(code)
        print(f"  生成: ecgformer_model.c")
    
    def _generate_fused_attention_code(self, block):
        """生成融合注意力块的代码
        
        融合操作序列: BATCH_MATMUL(QK) -> TRANSPOSE -> SOFTMAX -> BATCH_MATMUL(AV)
        
        策略: 逐Head计算，避免存储完整的 (8, 187, 187) 注意力矩阵
        每个Head只需要 187*187 = ~35KB 的临时空间
        
        Args:
            block: 注意力块信息字典
        """
        start_op = block['start_op']
        end_op = block['end_op']
        num_heads = block['num_heads']
        seq_len = block['seq_len']
        q_tid = block['q_tid']
        k_tid = block['k_tid']
        v_tid = block['v_tid']
        output_tid = block['output_tid']
        
        # 获取量化参数
        # Q: (1, 8, 187, 16), K: (1, 8, 16, 187), V: (1, 8, 187, 16)
        q_scale, q_zp = self._get_scale_zp(q_tid)
        k_scale, k_zp = self._get_scale_zp(k_tid)
        v_scale, v_zp = self._get_scale_zp(v_tid)
        
        # 获取中间张量的量化参数
        intermediate_tids = block['intermediate_tensors']
        qk_tid = intermediate_tids[0]  # BATCH_MATMUL output
        softmax_tid = intermediate_tids[2]  # SOFTMAX output
        
        qk_scale, qk_zp = self._get_scale_zp(qk_tid)
        softmax_scale, softmax_zp = self._get_scale_zp(softmax_tid)
        
        out_scale, out_zp = self._get_scale_zp(output_tid)
        
        # Q16 scale 计算
        # QK matmul: scale_qk = (q_scale * k_scale / qk_scale) * (1<<16)
        scale_qk_q16 = int(round((q_scale * k_scale / qk_scale) * (1 << 16)))
        
        # Attn*V matmul: scale_av = (softmax_scale * v_scale / out_scale) * (1<<16)
        scale_av_q16 = int(round((softmax_scale * v_scale / out_scale) * (1 << 16)))
        
        # 输出张量的内存分配
        memory_mode = getattr(self, 'memory_mode', 'static')
        alloc_code = ''
        if memory_mode in ('pingpong', 'reuse') and hasattr(self, 'slot_assignments'):
            if output_tid in self.slot_assignments:
                slot_idx = self.slot_assignments[output_tid]
                alloc_code = f'    *(ptensors + {output_tid}) = g_activation_pool + g_slot_offsets[{slot_idx}];\n'
        
        # 使用 V 张量的形状确定 head_dim
        v_info = self.tensors_info.get(v_tid, {})
        v_shape = v_info.get('shape', (1, num_heads, seq_len, 16))
        head_dim = int(v_shape[-1]) if len(v_shape) > 3 else 16
        
        code = f'''    // ========== 融合注意力块 (Ops #{start_op}-#{end_op}) ==========
    // 策略: 逐Head计算，避免存储完整的 (8, 187, 187) 注意力矩阵
    // 每个Head只需 ~{seq_len * seq_len} bytes 临时空间
{alloc_code}
    op_fused_attention_per_head(
        *(ptensors + {q_tid}),   // Q: (1, {num_heads}, {seq_len}, {head_dim})
        *(ptensors + {k_tid}),   // K: (1, {num_heads}, {head_dim}, {seq_len})
        *(ptensors + {v_tid}),   // V: (1, {num_heads}, {seq_len}, {head_dim})
        *(ptensors + {output_tid}),  // Out: (1, {num_heads}, {seq_len}, {head_dim})
        {num_heads}, {seq_len}, {head_dim},
        {scale_qk_q16}, {q_zp}, {k_zp}, {qk_zp},  // QK 量化参数
        {qk_scale:.10e}f, {softmax_scale:.10e}f, {softmax_zp},  // Softmax 量化参数
        {scale_av_q16}, {v_zp}, {out_zp}  // AV 量化参数
    );
    
'''
        return code

    def _generate_op_code_modular(self, op):
        """为单个操作生成代码 - Q16 Fixed-Point Style (模块化版本)
        使用ptensors指针数组, 预计算Q16定点scale, 适配INT26中间结果
        """
        op_id = op['id']
        op_type = op['type']
        inputs = op['inputs']
        outputs = op['outputs']
        
        out_tid = outputs[0]
        out_info = self.tensors_info.get(out_tid, {})
        out_size = out_info.get('size', 1)
        out_shape = out_info.get('shape', ())
        out_scale, out_zp = self._get_scale_zp(out_tid)
        
        memory_mode = getattr(self, 'memory_mode', 'static')
        
        code = f'    // Op#{op_id}: {op_type}\n'
        
        # 检查这个操作是否是融合注意力块的一部分
        is_fused = hasattr(self, 'fused_ops') and op_id in self.fused_ops
        if is_fused:
            # 检查是否是注意力块的起始操作
            for block in getattr(self, 'attention_blocks', []):
                if block['start_op'] == op_id:
                    # 生成融合注意力计算代码
                    return self._generate_fused_attention_code(block)
            # 如果不是起始操作，跳过（已在融合代码中处理）
            return f'    // Op#{op_id}: {op_type} (fused in attention block)\n'
        
        # 检查是否是零拷贝操作
        zero_copy_aliases = getattr(self, 'zero_copy_aliases', {})
        is_zero_copy = out_tid in zero_copy_aliases
        
        # 根据内存模式设置输出张量指针
        if memory_mode == 'pingpong':
            # pingpong模式使用slot分配 (与reuse模式相同的分配方式)
            if is_zero_copy:
                # 零拷贝: 输出直接使用输入的内存地址
                source_tid = zero_copy_aliases[out_tid]
                # 递归查找最终源
                while source_tid in zero_copy_aliases:
                    source_tid = zero_copy_aliases[source_tid]
                code += f'    *(ptensors + {out_tid}) = *(ptensors + {source_tid});  // zero-copy alias\n'
            elif hasattr(self, 'slot_assignments') and out_tid in self.slot_assignments:
                slot_idx = self.slot_assignments[out_tid]
                code += f'    *(ptensors + {out_tid}) = g_activation_pool + g_slot_offsets[{slot_idx}];\n'
            elif out_tid not in getattr(self, 'attention_tensors', set()):
                # 跳过中间注意力张量（不需要分配）
                code += f'    // Warning: tensor {out_tid} not assigned to slot\n'
        elif memory_mode == 'reuse' and hasattr(self, 'slot_assignments'):
            if is_zero_copy:
                source_tid = zero_copy_aliases[out_tid]
                while source_tid in zero_copy_aliases:
                    source_tid = zero_copy_aliases[source_tid]
                code += f'    *(ptensors + {out_tid}) = *(ptensors + {source_tid});  // zero-copy alias\n'
            elif out_tid in self.slot_assignments:
                slot_idx = self.slot_assignments[out_tid]
                code += f'    *(ptensors + {out_tid}) = g_activation_pool + g_slot_offsets[{slot_idx}];\n'
        
        if op_type == 'RESHAPE':
            if is_zero_copy:
                code += f'    // zero-copy: skip memcpy for RESHAPE\n'
            else:
                in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
                code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {in_size});\n'
        
        elif op_type == 'EXPAND_DIMS':
            if is_zero_copy:
                code += f'    // zero-copy: skip memcpy for EXPAND_DIMS\n'
            else:
                in_size = self.tensors_info.get(inputs[0], {}).get('size', 1)
                code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {in_size});\n'
        
        elif op_type == 'TRANSPOSE':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            perm_tid = inputs[1]
            perm = self.weights_data.get(perm_tid, np.arange(len(in_shape))).flatten().tolist()
            
            if len(in_shape) == 3:
                code += f'    op_transpose_3d(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])});\n'
            elif len(in_shape) == 4:
                code += f'    op_transpose_4d(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), '
                code += f'{in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]}, '
                code += f'{int(perm[0])}, {int(perm[1])}, {int(perm[2])}, {int(perm[3])});\n'
            else:
                code += f'    op_copy(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {out_size});\n'
        
        elif op_type == 'ADD':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            # Q16: scale1 = (s1/so) * (1<<16), scale2 = (s2/so) * (1<<16)
            scale1_q16 = int(round((s1 / out_scale) * (1 << 16)))
            scale2_q16 = int(round((s2 / out_scale) * (1 << 16)))
            code += f'    op_add(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'           {scale1_q16}, {z1}, {scale2_q16}, {z2}, {out_zp});\n'
        
        elif op_type == 'SUB':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(*(ptensors + {out_tid}), {out_zp}, {out_size});\n'
            else:
                scale1_q16 = int(round((s1 / out_scale) * (1 << 16)))
                scale2_q16 = int(round((s2 / out_scale) * (1 << 16)))
                code += f'    op_sub(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
                code += f'           {scale1_q16}, {z1}, {scale2_q16}, {z2}, {out_zp});\n'
        
        elif op_type == 'MUL':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            # Q16: scale = (s1 * s2 / so) * (1<<16)
            scale_q16 = int(round((s1 * s2 / out_scale) * (1 << 16)))
            code += f'    op_mul(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'           {scale_q16}, {z1}, {z2}, {out_zp});\n'
        
        elif op_type == 'SQUARED_DIFFERENCE':
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            if inputs[0] == inputs[1]:
                code += f'    memset(*(ptensors + {out_tid}), {out_zp}, {out_size});\n'
            else:
                # Q16: scale_diff = (s2/s1) * (1<<16), scale_out = (s1*s1/so) * (1<<16)
                scale_diff_q16 = int(round((s2 / s1) * (1 << 16)))
                scale_out_q16 = int(round((s1 * s1 / out_scale) * (1 << 16)))
                code += f'    op_squared_diff(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}), {out_size},\n'
                code += f'                    {scale_diff_q16}, {z1}, {z2}, {scale_out_q16}, {out_zp});\n'
        
        elif op_type == 'RSQRT':
            si, zi = self._get_scale_zp(inputs[0])
            # rsqrt使用浮点: scale_in = si, scale_out = so
            code += f'    op_rsqrt(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {out_size},\n'
            code += f'             {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        elif op_type == 'FULLY_CONNECTED':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            chn_in = in_shape[-1] if len(in_shape) > 0 else 1
            nbatch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            chn_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            pbias_str = f'(const int32_t*)*(ptensors + {inputs[2]})' if has_bias else 'NULL'
            
            # 检查是否有预计算的 Q16 scales
            if op_id in self.q16_scales:
                # 使用预计算的 Q16 scale 数组
                pscales_str = f'pscales_q16_op{op_id}'
            else:
                # 单通道 scale: 预计算单个 Q16 值
                weight_info = self.tensors_info.get(weight_tid, {})
                weight_scales = weight_info.get('scales', np.array([]))
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                scale_q16 = int(round((si * single_scale / out_scale) * (1 << 16)))
                code += f'    {{ static const int32_t pws_q16[1] = {{{scale_q16}}};\n'
                pscales_str = 'pws_q16'
            
            code += f'    op_fc(*(ptensors + {inputs[0]}), {nbatch}, {chn_in}, {chn_out},\n'
            code += f'          (const int8_t*)*(ptensors + {weight_tid}), {pbias_str}, *(ptensors + {out_tid}),\n'
            code += f'          {zi}, {pscales_str}, {out_zp});\n'
            
            if op_id not in self.q16_scales:
                code += '    }\n'
        
        elif op_type == 'CONV_2D':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            weight_tid = inputs[1]
            weight_shape = self.tensors_info.get(weight_tid, {}).get('shape', ())
            
            si, zi = self._get_scale_zp(inputs[0])
            chn_in = in_shape[-1] if len(in_shape) > 0 else 1
            spatial = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            chn_out = weight_shape[0] if len(weight_shape) > 0 else 1
            
            has_bias = len(inputs) > 2
            pbias_str = f'(const int32_t*)*(ptensors + {inputs[2]})' if has_bias else 'NULL'
            
            # 检查是否有预计算的 Q16 scales
            if op_id in self.q16_scales:
                pscales_str = f'pscales_q16_op{op_id}'
            else:
                weight_info = self.tensors_info.get(weight_tid, {})
                weight_scales = weight_info.get('scales', np.array([]))
                single_scale = weight_scales[0] if len(weight_scales) > 0 else 1.0
                scale_q16 = int(round((si * single_scale / out_scale) * (1 << 16)))
                code += f'    {{ static const int32_t pws_q16[1] = {{{scale_q16}}};\n'
                pscales_str = 'pws_q16'
            
            code += f'    op_fc(*(ptensors + {inputs[0]}), {spatial}, {chn_in}, {chn_out},\n'
            code += f'          (const int8_t*)*(ptensors + {weight_tid}), {pbias_str}, *(ptensors + {out_tid}),\n'
            code += f'          {zi}, {pscales_str}, {out_zp});\n'
            
            if op_id not in self.q16_scales:
                code += '    }\n'
        
        elif op_type == 'BATCH_MATMUL':
            in1_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            in2_shape = self.tensors_info.get(inputs[1], {}).get('shape', ())
            s1, z1 = self._get_scale_zp(inputs[0])
            s2, z2 = self._get_scale_zp(inputs[1])
            
            nbatch = in1_shape[0] if len(in1_shape) > 2 else 1
            side_m = in1_shape[1] if len(in1_shape) > 1 else 1
            side_k = in1_shape[2] if len(in1_shape) > 2 else in1_shape[1] if len(in1_shape) > 1 else 1
            side_n = in2_shape[2] if len(in2_shape) > 2 else in2_shape[1] if len(in2_shape) > 1 else 1
            
            # Q16: scale = (s1 * s2 / so) * (1<<16)
            scale_q16 = int(round((s1 * s2 / out_scale) * (1 << 16)))
            code += f'    op_batch_matmul(*(ptensors + {inputs[0]}), *(ptensors + {inputs[1]}), *(ptensors + {out_tid}),\n'
            code += f'                    {nbatch}, {side_m}, {side_k}, {side_n},\n'
            code += f'                    {scale_q16}, {z1}, {z2}, {out_zp});\n'
        
        elif op_type == 'MEAN':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            
            outer = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            reduce_len = in_shape[-1] if len(in_shape) > 0 else 1
            inner = 1
            
            # Q16: scale = (si / so) * (1<<16)
            scale_q16 = int(round((si / out_scale) * (1 << 16)))
            code += f'    op_mean(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}),\n'
            code += f'            {outer}, {reduce_len}, {inner},\n'
            code += f'            {scale_q16}, {zi}, {out_zp});\n'
        
        elif op_type == 'SOFTMAX':
            in_shape = self.tensors_info.get(inputs[0], {}).get('shape', ())
            si, zi = self._get_scale_zp(inputs[0])
            nbatch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            nclass = in_shape[-1] if len(in_shape) > 0 else 1
            
            # softmax使用浮点: scale_in = si, scale_out = so
            code += f'    op_softmax(*(ptensors + {inputs[0]}), *(ptensors + {out_tid}), {nbatch}, {nclass},\n'
            code += f'               {si:.10e}f, {zi}, {out_scale:.10e}f, {out_zp});\n'
        
        else:
            code += f'    // TODO: Implement {op_type}\n'
        
        # pingpong模式现在使用slot分配，不需要缓冲区切换
        
        code += '\n'
        return code


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成ECGformer C实现代码 (内存优化版，峰值152KB)')
    parser.add_argument('--memory-limit', type=int, default=262144,
                        help='内存限制(字节)，默认262144 (256KB)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出目录路径')
    args = parser.parse_args()
    
    model_path = os.path.join(PROJECT_ROOT, 'exported_models', 'tflite', 
                              'ecgformer_custom_ln_int8.tflite')
    
    print("="*60)
    print("生成ECGformer C实现 (内存优化版)")
    print("="*60)
    
    generator = ECGformerCGenerator(model_path)
    
    output_dir = args.output or os.path.join(SCRIPT_DIR, 'c_export_modular')
    print(f"\n输出目录: {output_dir}")
    print(f"内存限制: {args.memory_limit} bytes ({args.memory_limit/1024:.1f} KB)\n")
    generator.generate(output_dir, memory_limit=args.memory_limit)
