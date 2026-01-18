# -*- coding: utf-8 -*-
"""
评估TFLite模型、NumPy整数实现和C实现在测试集上的准确率
对比三个实现的一致性和性能

支持多线程评估以加速处理
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import time
import gc
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

try:
    import tensorflow as tf
except ImportError:
    print("警告: TensorFlow未安装，无法运行TFLite模型")
    tf = None
from integer_only_model import IntegerOnlyECGformer

# C实现
try:
    from verify_c_impl import ECGformerC, compile_c_library, print_c_implementation_metrics
    HAS_C_IMPL = True
except Exception as e:
    print(f"警告: C实现未就绪 - {e}")
    HAS_C_IMPL = False


# 类别映射
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_FULL_NAMES = {
    0: 'N (Normal beat)',
    1: 'S (Supraventricular premature beat)',
    2: 'V (Premature ventricular contraction)',
    3: 'F (Fusion of ventricular and normal beat)',
    4: 'Q (Unclassifiable beat)'
}


class TFLiteRunner:
    """TFLite模型运行器"""
    
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 检查是否是量化模型
        self.is_quantized = self.input_details[0]['dtype'] != np.float32
        if self.is_quantized:
            self.input_scale = self.input_details[0]['quantization'][0]
            self.input_zero_point = self.input_details[0]['quantization'][1]
            self.output_scale = self.output_details[0]['quantization'][0]
            self.output_zero_point = self.output_details[0]['quantization'][1]
    
    def predict_class(self, input_data: np.ndarray) -> int:
        """运行推理并返回类别"""
        # 量化输入
        if self.is_quantized:
            quantized = np.round(input_data / self.input_scale) + self.input_zero_point
            input_data = np.clip(quantized, -128, 127).astype(np.int8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return int(np.argmax(output))
    
    def get_int8_output(self, input_data: np.ndarray) -> np.ndarray:
        """运行推理并返回INT8输出"""
        if self.is_quantized:
            quantized = np.round(input_data / self.input_scale) + self.input_zero_point
            input_data = np.clip(quantized, -128, 127).astype(np.int8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index']).copy()


def load_test_data(csv_path: str, max_samples: int = None):
    """加载测试数据"""
    print(f"正在加载测试数据: {csv_path}")
    
    data = pd.read_csv(csv_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype(int).values
    
    if max_samples is not None and len(y) > max_samples:
        # 随机采样
        np.random.seed(42)
        indices = np.random.choice(len(y), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"测试集大小: {len(y)} 样本")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 调整形状
    X = X.reshape((X.shape[0], X.shape[1], 1)).astype(np.float32)
    
    return X, y


def evaluate_model(name: str, predictor_fn, X_test: np.ndarray, y_test: np.ndarray,
                   show_progress: bool = True) -> dict:
    """通用模型评估函数（单线程）"""
    print(f"\n{'='*60}")
    print(f"评估: {name} (单线程)")
    print('='*60)
    
    y_pred_all = []
    total_samples = len(y_test)
    
    start_time = time.time()
    
    for i in range(total_samples):
        X_single = X_test[i:i+1]
        y_pred = predictor_fn(X_single)
        y_pred_all.append(y_pred)
        
        if show_progress and (i + 1) % 1000 == 0:
            progress = (i + 1) / total_samples * 100
            print(f"\r进度: {i+1}/{total_samples} ({progress:.1f}%)", end='', flush=True)
    
    inference_time = time.time() - start_time
    if show_progress:
        print()
    
    y_pred = np.array(y_pred_all)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n推理时间: {inference_time:.2f} 秒")
    print(f"平均每个样本: {inference_time/total_samples*1000:.3f} 毫秒")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'name': name,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'inference_time': inference_time,
        'samples_per_second': total_samples / inference_time
    }


def _worker_batch_predict(batch_data, model_path, impl_type):
    """工作进程的批处理预测函数
    
    注意：每个进程会独立加载模型，所以批次不宜太小，否则模型加载开销会很大
    返回：(indices, predictions, load_time, inference_time)
    """
    import time as _time
    indices, X_batch = batch_data
    
    if impl_type == 'tflite':
        import tensorflow as tf
        load_start = _time.time()
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        is_quantized = input_details[0]['dtype'] != np.float32
        if is_quantized:
            input_scale = input_details[0]['quantization'][0]
            input_zp = input_details[0]['quantization'][1]
        load_time = _time.time() - load_start
        
        infer_start = _time.time()
        predictions = []
        for x in X_batch:
            x_input = x.reshape(1, -1, 1)
            if is_quantized:
                quantized = np.round(x_input / input_scale) + input_zp
                x_input = np.clip(quantized, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], x_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(int(np.argmax(output)))
        infer_time = _time.time() - infer_start
        
        return indices, predictions, load_time, infer_time
    
    elif impl_type == 'numpy':
        from integer_only_model import IntegerOnlyECGformer
        # 模型加载开销较大，所以在worker内只加载一次
        load_start = _time.time()
        model = IntegerOnlyECGformer(model_path)
        load_time = _time.time() - load_start
        
        infer_start = _time.time()
        predictions = []
        for x in X_batch:
            x_input = x.reshape(1, -1, 1).astype(np.float32)
            output = model.forward(x_input)
            predictions.append(int(np.argmax(output)))
        infer_time = _time.time() - infer_start
        
        return indices, predictions, load_time, infer_time
    
    elif impl_type == 'c':
        from verify_c_impl import ECGformerC
        load_start = _time.time()
        model = ECGformerC()
        load_time = _time.time() - load_start
        
        infer_start = _time.time()
        predictions = []
        for x in X_batch:
            x_input = x.reshape(1, -1, 1).astype(np.float32)
            pred, _ = model.inference(x_input)
            predictions.append(pred)
        infer_time = _time.time() - infer_start
        
        return indices, predictions, load_time, infer_time
    
    return indices, [], 0, 0


def evaluate_model_parallel(name: str, impl_type: str, model_path: str,
                           X_test: np.ndarray, y_test: np.ndarray,
                           num_workers: int = None,
                           show_progress: bool = True) -> dict:
    """多进程并行评估模型
    
    注意：对于NumPy实现，每个进程需要加载模型（解析TFLite文件），
    所以批次数应该等于进程数，避免重复加载模型的开销。
    """
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    print(f"\n{'='*60}")
    print(f"评估: {name} (多进程, {num_workers}个工作进程)")
    print('='*60)
    
    total_samples = len(y_test)
    
    # 将数据均匀分成恰好 num_workers 个批次
    # 使用 np.array_split 可以确保批次数等于 num_workers
    indices_all = np.arange(total_samples)
    indices_splits = np.array_split(indices_all, num_workers)
    X_splits = np.array_split(X_test, num_workers)
    
    batches = []
    for idx_split, X_split in zip(indices_splits, X_splits):
        if len(idx_split) > 0:
            # 预处理：去掉最后一维，因为worker会重新reshape
            X_batch_flat = X_split.reshape(X_split.shape[0], -1)
            batches.append((idx_split.tolist(), X_batch_flat))
    
    batch_sizes = [len(b[0]) for b in batches]
    print(f"数据分成 {len(batches)} 个批次，每批 {min(batch_sizes)}~{max(batch_sizes)} 个样本")
    
    start_time = time.time()
    
    # 使用进程池并行处理
    y_pred_all = [None] * total_samples
    worker_fn = partial(_worker_batch_predict, model_path=model_path, impl_type=impl_type)
    
    completed = 0
    total_load_time = 0
    total_infer_time = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, batch): batch for batch in batches}
        
        for future in futures:
            indices, predictions, load_time, infer_time = future.result()
            total_load_time = max(total_load_time, load_time)  # 并行加载，取最大值
            total_infer_time = max(total_infer_time, infer_time)  # 并行推理，取最大值
            
            for idx, pred in zip(indices, predictions):
                y_pred_all[idx] = pred
            completed += len(indices)
            
            if show_progress:
                progress = completed / total_samples * 100
                print(f"\r进度: {completed}/{total_samples} ({progress:.1f}%)", end='', flush=True)
    
    wall_time = time.time() - start_time
    if show_progress:
        print()
    
    y_pred = np.array(y_pred_all)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n总耗时: {wall_time:.2f} 秒 (模型加载: ~{total_load_time:.2f}s, 推理: ~{total_infer_time:.2f}s)")
    print(f"等效单线程推理时间: {total_infer_time * num_workers:.2f} 秒")
    print(f"吞吐量: {total_samples/wall_time:.1f} 样本/秒 (纯推理: {total_samples/total_infer_time:.1f} 样本/秒)")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'name': name,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'inference_time': wall_time,
        'samples_per_second': total_samples / wall_time
    }


def print_classification_report(y_test: np.ndarray, y_pred: np.ndarray, name: str):
    """打印分类报告"""
    print(f"\n{name} 分类报告:")
    print("-" * 60)
    # 使用labels参数确保所有类别都包含在报告中
    labels = list(range(len(CLASS_NAMES)))
    print(classification_report(y_test, y_pred, labels=labels, 
                               target_names=CLASS_NAMES, digits=4, zero_division=0))


def compare_implementations(results: list, y_test: np.ndarray):
    """比较多个实现的结果"""
    print("\n" + "=" * 60)
    print("实现对比")
    print("=" * 60)
    
    # 准确率对比
    print("\n准确率对比:")
    print("-" * 40)
    for r in results:
        print(f"  {r['name']:<25}: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
    
    # 速度对比
    print("\n速度对比 (样本/秒):")
    print("-" * 40)
    for r in results:
        print(f"  {r['name']:<25}: {r['samples_per_second']:.1f}")
    
    # 预测一致性
    if len(results) >= 2:
        print("\n预测一致性:")
        print("-" * 40)
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                agreement = np.mean(results[i]['y_pred'] == results[j]['y_pred'])
                print(f"  {results[i]['name']} vs {results[j]['name']}: {agreement:.4f} ({agreement*100:.2f}%)")


def main():
    # 模型路径
    model_path = os.path.join(PROJECT_ROOT, 'exported_models', 'tflite',
                              'ecgformer_custom_ln_int8.tflite')
    test_csv = os.path.join(PROJECT_ROOT, 'dataset', 'mitbih_test.csv')
    
    # 检查文件存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return
    if not os.path.exists(test_csv):
        print(f"错误: 测试数据不存在 - {test_csv}")
        return
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='评估ECGformer模型实现')
    parser.add_argument('--no-parallel', '-np', action='store_true', 
                        help='禁用多进程并行评估 (默认启用并行)')
    parser.add_argument('--workers', '-w', type=int, default=16,
                        help='并行工作进程数 (默认: 16)')
    parser.add_argument('--samples', '-n', type=int, default=None,
                        help='评估样本数 (默认: 全部)')
    #添加评估方法选择，允许选择多种方法，如--method tflite numpy c
    parser.add_argument('--method', '-m', nargs='+', choices=['tflite', 'numpy', 'c'],
                        default=['tflite', 'c'],
                        help='选择要评估的实现方法 (默认: tflite, c)')
    parser.add_argument('--metrics', action='store_true',
                        help='显示C实现的硬件相关指标')
    parser.add_argument('--metrics-only', action='store_true',
                        help='仅显示C实现的硬件相关指标，不运行评估')
    args = parser.parse_args()
    
    # 仅显示指标模式
    if args.metrics_only:
        if HAS_C_IMPL:
            print_c_implementation_metrics()
        else:
            print("错误: C实现不可用，无法显示指标")
        return
    
    # 默认启用并行，除非指定--no-parallel
    args.parallel = not args.no_parallel
    
    # 加载测试数据
    X_test, y_test = load_test_data(test_csv, max_samples=args.samples)
    
    results = []
    
    if args.parallel:
        print("\n" + "#" * 60)
        print("# 多进程并行评估模式")
        print("#" * 60)
        
        num_workers = args.workers
        print(f"使用 {num_workers} 个工作进程")
        
        if 'tflite' in args.method:
            # 1. 评估TFLite模型
            print("\n" + "#" * 60)
            print("# 1. TFLite 模型 (多进程)")
            print("#" * 60)
            tflite_result = evaluate_model_parallel(
                "TFLite INT8", 'tflite', model_path, X_test, y_test, num_workers)
            results.append(tflite_result)

        if 'numpy' in args.method:
            # 2. 评估NumPy整数实现
            print("\n" + "#" * 60)
            print("# 2. NumPy 整数实现 (多进程)")
            print("#" * 60)
            numpy_result = evaluate_model_parallel(
                "NumPy Integer", 'numpy', model_path, X_test, y_test, num_workers)
            results.append(numpy_result)
        
        # 3. 评估C实现
        if HAS_C_IMPL and 'c' in args.method:
            print("\n" + "#" * 60)
            print("# 3. C 实现 (多进程)")
            print("#" * 60)
            try:
                compile_c_library()
                c_result = evaluate_model_parallel(
                    "C Implementation", 'c', model_path, X_test, y_test, num_workers)
                results.append(c_result)
            except Exception as e:
                print(f"C实现评估失败: {e}")
    else:
        if 'tflite' in args.method:
            # 单线程评估
            # 1. 评估TFLite模型
            print("\n" + "#" * 60)
            print("# 1. TFLite 模型")
            print("#" * 60)
            
            tflite_runner = TFLiteRunner(model_path)
            
            def tflite_predict(x):
                return tflite_runner.predict_class(x)
            
            tflite_result = evaluate_model("TFLite INT8", tflite_predict, X_test, y_test)
            results.append(tflite_result)
        
        if 'numpy' in args.method:
            # 2. 评估NumPy整数实现
            print("\n" + "#" * 60)
            print("# 2. NumPy 整数实现")
            print("#" * 60)
            
            numpy_model = IntegerOnlyECGformer(model_path)
            
            def numpy_predict(x):
                output = numpy_model.forward(x)
                return int(np.argmax(output))
            
            numpy_result = evaluate_model("NumPy Integer", numpy_predict, X_test, y_test)
            results.append(numpy_result)
        
        # 3. 评估C实现
        if HAS_C_IMPL and 'c' in args.method:
            print("\n" + "#" * 60)
            print("# 3. C 实现")
            print("#" * 60)
            
            try:
                compile_c_library()
                c_model = ECGformerC()
                
                def c_predict(x):
                    pred, _ = c_model.inference(x)
                    return pred
                
                c_result = evaluate_model("C Implementation", c_predict, X_test, y_test)
                results.append(c_result)
            except Exception as e:
                print(f"C实现评估失败: {e}")
    
    # 打印分类报告
    print_classification_report(y_test, results[0]['y_pred'], results[0]['name'])
    
    # 对比结果
    compare_implementations(results, y_test)
    
    # 如果包含C实现且指定了--metrics，打印硬件指标
    if HAS_C_IMPL and (args.metrics or 'c' in args.method):
        print_c_implementation_metrics()
    
    # 保存结果
    output_file = os.path.join(PROJECT_ROOT, 'results', 'accuracy_comparison.txt')
    with open(output_file, 'w') as f:
        f.write("ECGformer 实现对比报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 准确率对比:\n")
        for r in results:
            f.write(f"   {r['name']:<25}: {r['accuracy']:.4f}\n")
        
        f.write("\n2. 速度对比 (样本/秒):\n")
        for r in results:
            f.write(f"   {r['name']:<25}: {r['samples_per_second']:.1f}\n")
        
        f.write("\n3. 分类报告:\n")
        labels = list(range(len(CLASS_NAMES)))
        f.write(classification_report(y_test, results[0]['y_pred'], 
                                      labels=labels, target_names=CLASS_NAMES, 
                                      digits=4, zero_division=0))
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
