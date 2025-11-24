"""
Keras 与 TFLite 模型推理对比工具

功能：
1. 从测试集中抽取指定数量的样本
2. 分别使用 Keras 和 TFLite 模型进行推理
3. 展示每个样本的预测结果和概率输出
4. 计算两个模型输出的差异
5. 统计分类一致性

使用方法:
    python compare_inference.py --num_samples 20
    python compare_inference.py --num_samples 50 --random_seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置
KERAS_MODEL_PATH = './ckpts/best_model.keras'
TFLITE_MODEL_PATH = './exported_models/tflite/ecgformer_ptq_dynamic.tflite'
TEST_DATA_PATH = './dataset/mitbih_test.csv'

# 类别标签
CLASS_LABELS = {
    0: 'Normal (正常)',
    1: 'Supraventricular (室上性)',
    2: 'Ventricular (室性)',
    3: 'Fusion (融合)',
    4: 'Unknown (未知)'
}

# 禁用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_data(num_samples, random_seed=None):
    """加载并预处理数据"""
    print(f"\n{'='*80}")
    print("加载数据")
    print('='*80)
    
    # 读取数据
    data = pd.read_csv(TEST_DATA_PATH, header=None)
    y = data.iloc[:, -1].astype(int).to_numpy()
    x = data.iloc[:, :-1]
    
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    
    # 抽取样本
    if random_seed is not None:
        np.random.seed(random_seed)
        indices = np.random.choice(len(x), size=min(num_samples, len(x)), replace=False)
    else:
        indices = np.arange(min(num_samples, len(x)))
    
    x_sampled = x[indices]
    y_sampled = y[indices]
    
    print(f"✓ 总数据量: {len(x)}")
    print(f"✓ 抽取样本: {len(x_sampled)}")
    print(f"✓ 数据形状: {x_sampled.shape}")
    if random_seed is not None:
        print(f"✓ 随机种子: {random_seed}")
    
    # 统计类别分布
    unique, counts = np.unique(y_sampled, return_counts=True)
    print(f"\n类别分布:")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls} ({CLASS_LABELS[cls]}): {cnt} 样本")
    
    return x_sampled, y_sampled, indices


def keras_inference(model_path, x):
    """Keras 模型推理"""
    print(f"\n{'='*80}")
    print("Keras 模型推理")
    print('='*80)
    
    model = tf.keras.models.load_model(model_path)
    print(f"✓ 模型加载成功: {model_path}")
    
    # 推理
    predictions = model.predict(x, verbose=0)
    print(f"✓ 推理完成: {len(x)} 个样本")
    
    return predictions


def tflite_inference(model_path, x):
    """TFLite 模型推理"""
    print(f"\n{'='*80}")
    print("TFLite 模型推理")
    print('='*80)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✓ 模型加载成功: {model_path}")
    print(f"  输入形状: {input_details[0]['shape']}")
    print(f"  输出形状: {output_details[0]['shape']}")
    
    # 逐个样本推理
    predictions = []
    for i in range(len(x)):
        sample = x[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output)
    
    predictions = np.concatenate(predictions, axis=0)
    print(f"✓ 推理完成: {len(x)} 个样本")
    
    return predictions


def analyze_results(keras_pred, tflite_pred, y_true, indices):
    """分析和展示结果"""
    print(f"\n{'='*80}")
    print("推理结果对比")
    print('='*80)
    
    # 获取预测类别
    keras_class = np.argmax(keras_pred, axis=1)
    tflite_class = np.argmax(tflite_pred, axis=1)
    
    # 计算差异
    diff = keras_pred - tflite_pred
    abs_diff = np.abs(diff)
    
    # 构建结果表格
    results = []
    for i in range(len(keras_pred)):
        row = {
            '样本': f"#{indices[i]}",
            '真实类别': f"{y_true[i]} ({CLASS_LABELS[y_true[i]].split()[0]})",
            'Keras预测': f"{keras_class[i]} ({CLASS_LABELS[keras_class[i]].split()[0]})",
            'TFLite预测': f"{tflite_class[i]} ({CLASS_LABELS[tflite_class[i]].split()[0]})",
            'Keras置信度': f"{keras_pred[i, keras_class[i]]:.4f}",
            'TFLite置信度': f"{tflite_pred[i, tflite_class[i]]:.4f}",
            '最大差异': f"{np.max(abs_diff[i]):.6f}",
            '一致': '✓' if keras_class[i] == tflite_class[i] else '✗'
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    print(f"\n{tabulate(df, headers='keys', tablefmt='grid', showindex=False)}")
    
    return keras_class, tflite_class, diff, abs_diff


def show_probability_details(keras_pred, tflite_pred, y_true, keras_class, tflite_class, indices, num_show=5):
    """展示详细的概率输出"""
    print(f"\n{'='*80}")
    print(f"详细概率输出 (前 {num_show} 个样本)")
    print('='*80)
    
    for i in range(min(num_show, len(keras_pred))):
        print(f"\n样本 #{indices[i]} (真实类别: {y_true[i]} - {CLASS_LABELS[y_true[i]]})")
        print("-" * 80)
        
        # Keras 输出
        print(f"Keras 输出:")
        print(f"  预测类别: {keras_class[i]} ({CLASS_LABELS[keras_class[i]]})")
        print(f"  概率分布:")
        for cls in range(5):
            bar_len = int(keras_pred[i, cls] * 50)
            bar = '█' * bar_len
            print(f"    类别 {cls}: {keras_pred[i, cls]:.6f} {bar}")
        
        # TFLite 输出
        print(f"\nTFLite 输出:")
        print(f"  预测类别: {tflite_class[i]} ({CLASS_LABELS[tflite_class[i]]})")
        print(f"  概率分布:")
        for cls in range(5):
            bar_len = int(tflite_pred[i, cls] * 50)
            bar = '█' * bar_len
            print(f"    类别 {cls}: {tflite_pred[i, cls]:.6f} {bar}")
        
        # 差异
        diff = keras_pred[i] - tflite_pred[i]
        print(f"\n差异 (Keras - TFLite):")
        for cls in range(5):
            print(f"    类别 {cls}: {diff[cls]:+.6f}")


def calculate_statistics(keras_pred, tflite_pred, keras_class, tflite_class, y_true, diff):
    """计算统计信息"""
    print(f"\n{'='*80}")
    print("统计分析")
    print('='*80)
    
    # 分类一致性
    consistency = np.mean(keras_class == tflite_class) * 100
    print(f"\n【分类一致性】")
    print(f"  预测一致率: {consistency:.2f}% ({np.sum(keras_class == tflite_class)}/{len(keras_class)} 样本)")
    
    if consistency < 100:
        inconsistent = np.where(keras_class != tflite_class)[0]
        print(f"  不一致样本:")
        for idx in inconsistent:
            print(f"    样本 {idx}: Keras={keras_class[idx]}, TFLite={tflite_class[idx]}, 真实={y_true[idx]}")
    
    # 准确率
    keras_acc = np.mean(keras_class == y_true) * 100
    tflite_acc = np.mean(tflite_class == y_true) * 100
    print(f"\n【分类准确率】")
    print(f"  Keras 准确率:  {keras_acc:.2f}% ({np.sum(keras_class == y_true)}/{len(y_true)} 样本)")
    print(f"  TFLite 准确率: {tflite_acc:.2f}% ({np.sum(tflite_class == y_true)}/{len(y_true)} 样本)")
    print(f"  准确率差异:    {abs(keras_acc - tflite_acc):.2f}%")
    
    # 概率输出差异
    abs_diff = np.abs(diff)
    print(f"\n【概率输出差异】")
    print(f"  平均绝对差异 (MAE):  {np.mean(abs_diff):.6f}")
    print(f"  均方根差异 (RMSE):   {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"  最大绝对差异:        {np.max(abs_diff):.6f}")
    print(f"  最小绝对差异:        {np.min(abs_diff):.6f}")
    print(f"  标准差:              {np.std(abs_diff):.6f}")
    
    # 按类别分析差异
    print(f"\n【各类别平均差异】")
    for cls in range(5):
        cls_mask = y_true == cls
        if np.sum(cls_mask) > 0:
            cls_diff = np.mean(abs_diff[cls_mask])
            print(f"  类别 {cls} ({CLASS_LABELS[cls].split()[0]:15s}): {cls_diff:.6f} ({np.sum(cls_mask)} 样本)")
    
    # 余弦相似度
    keras_flat = keras_pred.flatten()
    tflite_flat = tflite_pred.flatten()
    cosine_sim = np.dot(keras_flat, tflite_flat) / (np.linalg.norm(keras_flat) * np.linalg.norm(tflite_flat))
    print(f"\n【整体相似度】")
    print(f"  余弦相似度: {cosine_sim:.8f}")
    
    # 置信度分析
    print(f"\n【置信度分析】")
    keras_conf = keras_pred[np.arange(len(keras_class)), keras_class]
    tflite_conf = tflite_pred[np.arange(len(tflite_class)), tflite_class]
    print(f"  Keras 平均置信度:  {np.mean(keras_conf):.4f} ± {np.std(keras_conf):.4f}")
    print(f"  TFLite 平均置信度: {np.mean(tflite_conf):.4f} ± {np.std(tflite_conf):.4f}")
    
    return {
        'consistency': consistency,
        'keras_acc': keras_acc,
        'tflite_acc': tflite_acc,
        'mae': np.mean(abs_diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'max_diff': np.max(abs_diff),
        'cosine_sim': cosine_sim
    }


def plot_visualizations(keras_pred, tflite_pred, y_true, keras_class, tflite_class, 
                        indices, diff, stats, output_dir='./results'):
    """生成可视化图表"""
    print(f"\n{'='*80}")
    print("生成可视化图表")
    print('='*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 限制显示样本数（避免图表过于拥挤）
    max_samples_display = min(20, len(keras_pred))
    
    # 1. 柱状对比图 - 显示前N个样本的5类概率对比
    print("\n生成柱状对比图...")
    fig, axes = plt.subplots(max_samples_display, 1, figsize=(14, max_samples_display * 1.5))
    if max_samples_display == 1:
        axes = [axes]
    
    for i in range(max_samples_display):
        ax = axes[i]
        x_pos = np.arange(5)
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, keras_pred[i], width, label='Keras', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, tflite_pred[i], width, label='TFLite', alpha=0.8, color='#A23B72')
        
        # 标注预测类别
        ax.axvline(keras_class[i], color='#2E86AB', linestyle='--', alpha=0.3, linewidth=2)
        ax.axvline(tflite_class[i], color='#A23B72', linestyle='--', alpha=0.3, linewidth=2)
        
        ax.set_ylabel('Probability', fontsize=9)
        ax.set_title(f'Sample#{indices[i]} (True: {y_true[i]}, KPred: {keras_class[i]}, TPred: {tflite_class[i]})', 
                     fontsize=10, pad=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Class {j}' for j in range(5)], fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'probability_comparison_bars.png')
    plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {bar_chart_path}")
    
    # 2. 热力图 - Keras vs TFLite 概率矩阵
    print("\n生成概率热力图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(keras_pred)*0.3)))
    
    # Keras 热力图
    sns.heatmap(keras_pred, annot=False, cmap='YlOrRd', ax=axes[0], 
                cbar_kws={'label': 'Probability'}, vmin=0, vmax=1)
    axes[0].set_title('Keras Model Output Probability', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=11)
    axes[0].set_ylabel('Sample Index', fontsize=11)
    axes[0].set_xticklabels([f'{i}' for i in range(5)])
    
    # TFLite 热力图
    sns.heatmap(tflite_pred, annot=False, cmap='YlOrRd', ax=axes[1], 
                cbar_kws={'label': 'Probability'}, vmin=0, vmax=1)
    axes[1].set_title('TFLite Model Output Probability', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=11)
    axes[1].set_ylabel('Sample Index', fontsize=11)
    axes[1].set_xticklabels([f'{i}' for i in range(5)])
    
    # 差异热力图
    abs_diff = np.abs(diff)
    sns.heatmap(abs_diff, annot=False, cmap='RdYlGn_r', ax=axes[2], 
                cbar_kws={'label': 'Absolute Difference'}, vmin=0, vmax=np.percentile(abs_diff, 95))
    axes[2].set_title('Absolute Difference in Probability', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Class', fontsize=11)
    axes[2].set_ylabel('Sample Index', fontsize=11)
    axes[2].set_xticklabels([f'{i}' for i in range(5)])
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'probability_heatmaps.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {heatmap_path}")
    
    # 3. 散点图 - Keras vs TFLite 直接对比
    print("\n生成散点对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 为每个类别绘制散点图
    for cls in range(5):
        row = cls // 3
        col = cls % 3
        ax = axes[row, col]
        
        keras_cls = keras_pred[:, cls]
        tflite_cls = tflite_pred[:, cls]
        
        # 按预测正确性着色
        correct_mask = (keras_class == y_true) & (tflite_class == y_true)
        
        ax.scatter(keras_cls[correct_mask], tflite_cls[correct_mask], 
                  alpha=0.6, s=30, c='#2E86AB', label='Correct Prediction', edgecolors='white', linewidth=0.5)
        ax.scatter(keras_cls[~correct_mask], tflite_cls[~correct_mask], 
                  alpha=0.6, s=30, c='#F18F01', label='Incorrect Prediction', edgecolors='white', linewidth=0.5)
        
        # 绘制y=x参考线
        max_val = max(keras_cls.max(), tflite_cls.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1.5, label='Perfect Agreement')
        
        ax.set_xlabel('Keras Probability', fontsize=10)
        ax.set_ylabel('TFLite Probability', fontsize=10)
        ax.set_title(f'Class {cls} ({CLASS_LABELS[cls].split()[0]})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        if cls == 0:
            ax.legend(fontsize=8, loc='upper left')
    
    # 删除多余的子图
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'probability_scatter.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {scatter_path}")
    
    # 4. 差异分布直方图
    print("\n生成差异分布图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 4.1 整体差异直方图
    ax = axes[0, 0]
    diff_flat = diff.flatten()
    ax.hist(diff_flat, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Difference')
    ax.set_xlabel('Difference (Keras - TFLite)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Overall Probability Difference Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y', linestyle=':')
    
    # 4.2 绝对差异直方图
    ax = axes[0, 1]
    abs_diff_flat = np.abs(diff_flat)
    ax.hist(abs_diff_flat, bins=50, alpha=0.7, color='#A23B72', edgecolor='black', linewidth=0.5)
    ax.axvline(stats['mae'], color='green', linestyle='--', linewidth=2, 
              label=f'Mean Absolute Error={stats["mae"]:.6f}')
    ax.set_xlabel('Absolute Difference |Keras - TFLite|', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Absolute Difference Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y', linestyle=':')
    
    # 4.3 Average Difference by Class
    ax = axes[1, 0]
    class_diffs = [np.mean(np.abs(diff[:, cls])) for cls in range(5)]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax.bar(range(5), class_diffs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Mean Absolute Difference', fontsize=11)
    ax.set_title('Average Probability Difference by Class', fontsize=12, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels([CLASS_LABELS[i].split()[0] for i in range(5)], rotation=15, ha='right')
    ax.grid(alpha=0.3, axis='y', linestyle=':')
    
    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars, class_diffs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 4.4 统计摘要文本
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Statistical Summary
    {'='*40}
    
    Classification Consistency:     {stats['consistency']:.2f}%
    
    Accuracy:
      Keras:        {stats['keras_acc']:.2f}%
      TFLite:       {stats['tflite_acc']:.2f}%
    
    Probability Differences:
      Mean Absolute Error:  {stats['mae']:.6f}
      Root Mean Square Error:    {stats['rmse']:.6f}
      Maximum Difference:      {stats['max_diff']:.6f}
    
    Similarity:
      Cosine Similarity:    {stats['cosine_sim']:.8f}
    
    Sample Size:       {len(keras_pred)}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'difference_distribution.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {dist_path}")
    
    # 5. 综合对比仪表盘（仅当样本数较少时）
    if len(keras_pred) <= 10:
        print("\n生成详细对比仪表盘...")
        fig = plt.figure(figsize=(18, len(keras_pred) * 2.5))
        
        for i in range(len(keras_pred)):
            # 左侧：柱状图
            ax1 = plt.subplot(len(keras_pred), 2, i*2 + 1)
            x_pos = np.arange(5)
            width = 0.35
            
            bars1 = ax1.bar(x_pos - width/2, keras_pred[i], width, label='Keras', 
                           alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=1)
            bars2 = ax1.bar(x_pos + width/2, tflite_pred[i], width, label='TFLite', 
                           alpha=0.8, color='#A23B72', edgecolor='black', linewidth=1)
            
            # 标注数值
            for bar in bars1:
                height = bar.get_height()
                if height > 0.01:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                if height > 0.01:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax1.set_ylabel('Probability', fontsize=10)
            ax1.set_title(f'Sample#{indices[i]} | True: {y_true[i]} ({CLASS_LABELS[y_true[i]].split()[0]}) | '
                         f'Prediction: K={keras_class[i]}, T={tflite_class[i]}', 
                         fontsize=11, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([CLASS_LABELS[j].split()[0] for j in range(5)], 
                               rotation=20, ha='right', fontsize=9)
            ax1.set_ylim([0, 1.1])
            ax1.grid(axis='y', alpha=0.3, linestyle=':')
            ax1.legend(loc='upper right', fontsize=9)
            
            # 右侧：差异图
            ax2 = plt.subplot(len(keras_pred), 2, i*2 + 2)
            diff_sample = diff[i]
            colors_diff = ['#C73E1D' if d < 0 else '#6A994E' for d in diff_sample]
            bars_diff = ax2.bar(x_pos, diff_sample, color=colors_diff, alpha=0.8, 
                               edgecolor='black', linewidth=1)
            
            # 标注数值
            for bar, val in zip(bars_diff, diff_sample):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:+.4f}', ha='center', 
                        va='bottom' if val >= 0 else 'top', fontsize=8)
            
            ax2.axhline(0, color='black', linewidth=1.5, linestyle='-')
            ax2.set_ylabel('Difference (K-T)', fontsize=10)
            ax2.set_title(f'Probability Difference | MAE={np.mean(np.abs(diff_sample)):.6f}', 
                         fontsize=11, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([CLASS_LABELS[j].split()[0] for j in range(5)], 
                               rotation=20, ha='right', fontsize=9)
            ax2.grid(axis='y', alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, 'detailed_comparison_dashboard.png')
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存至: {dashboard_path}")
    
    print(f"\n✓ 所有可视化图表已生成！")
    print(f"  图表保存目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Keras 与 TFLite 模型推理对比')
    parser.add_argument('--num_samples', type=int, default=20, 
                        help='抽取的样本数量 (默认: 20)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='随机种子，用于可重复采样 (默认: None，顺序采样)')
    parser.add_argument('--show_details', type=int, default=5,
                        help='展示详细概率输出的样本数 (默认: 5)')
    parser.add_argument('--keras_model', type=str, default=KERAS_MODEL_PATH,
                        help='Keras 模型路径')
    parser.add_argument('--tflite_model', type=str, default=TFLITE_MODEL_PATH,
                        help='TFLite 模型路径')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='图表输出目录 (默认: ./results)')
    args = parser.parse_args()
    
    print("="*80)
    print("Keras vs TFLite 模型推理对比工具")
    print("="*80)
    print(f"Keras 模型:  {args.keras_model}")
    print(f"TFLite 模型: {args.tflite_model}")
    print(f"样本数量:    {args.num_samples}")
    
    # 加载数据
    x_test, y_test, indices = load_data(args.num_samples, args.random_seed)
    
    # Keras 推理
    keras_pred = keras_inference(args.keras_model, x_test)
    
    # TFLite 推理
    tflite_pred = tflite_inference(args.tflite_model, x_test)
    
    # 分析结果
    keras_class, tflite_class, diff, abs_diff = analyze_results(
        keras_pred, tflite_pred, y_test, indices
    )
    
    # 展示详细概率输出
    show_probability_details(
        keras_pred, tflite_pred, y_test, keras_class, tflite_class, 
        indices, args.show_details
    )
    
    # 统计分析
    stats = calculate_statistics(
        keras_pred, tflite_pred, keras_class, tflite_class, y_test, diff
    )
    
    # 生成可视化图表
    if args.plot:
        plot_visualizations(
            keras_pred, tflite_pred, y_test, keras_class, tflite_class,
            indices, diff, stats, args.output_dir
        )
    
    print(f"\n{'='*80}")
    print("✓ 对比完成！")
    if args.plot:
        print(f"✓ 可视化图表已保存至: {args.output_dir}")
    print('='*80)


if __name__ == '__main__':
    main()
