#!/bin/bash
# ECGformer TFLite 量化 - 一键运行所有量化方法

echo "======================================"
echo "ECGformer TFLite 量化测试"
echo "======================================"
echo ""

# 创建输出目录
mkdir -p ./exported_models/tflite

# 1. 动态范围量化（最快）
echo "1. 执行动态范围量化 (Dynamic Range Quantization)..."
python quantize_to_tflite.py --method ptq --ptq_type dynamic --eval
echo ""

# 2. Float16量化
echo "2. 执行Float16量化..."
python quantize_to_tflite.py --method ptq --ptq_type float16 --eval
echo ""

# 3. 全整数量化
echo "3. 执行全整数量化 (Full Integer Quantization)..."
python quantize_to_tflite.py --method ptq --ptq_type int8 --eval
echo ""

# 4. 量化感知训练（可选，需要较长时间）
read -p "是否执行量化感知训练 (QAT)? 这需要较长时间 [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "4. 执行量化感知训练..."
    python quantize_to_tflite.py --method qat --epochs 10 --eval
    echo ""
fi

# 5. 测试所有生成的模型
echo "======================================"
echo "测试所有生成的TFLite模型"
echo "======================================"
python test_tflite_models.py

echo ""
echo "======================================"
echo "量化完成！"
echo "======================================"
echo "查看结果："
echo "  ls -lh ./exported_models/tflite/"

