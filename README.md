# 乳腺癌超声图像分类

这是一个基于 Vision Transformer (ViT) 的乳腺癌超声图像分类系统，用于对乳腺超声图像进行良恶性分类。

## 功能特点

- 使用 Vision Transformer 架构进行图像分类
- 支持三分类：良性(benign)、恶性(malignant)和正常(normal)
- 包含完整的数据预处理和增强流程
- 提供详细的训练和评估指标
- 支持模型训练过程可视化
- 包含混淆矩阵和预测结果可视化

## 项目结构

```
.
├── Dataset_BUSI_with_GT/    # 数据集目录
├── models.py                # 模型定义文件
├── train_and_evaluate.py    # 训练和评估脚本
├── results/                # 结果输出目录
├── trained_models/         # 训练好的模型保存目录
└── tensorboard/            # TensorBoard 日志目录
```

## 模型架构

本项目使用 Vision Transformer (ViT) 作为基础模型，主要特点：
- 图像分块嵌入
- 多头自注意力机制
- 位置编码
- 多层 Transformer 块
- 分类头

### Peek: 模型改动

![Screenshot 2025-05-25 at 23.13.12](../../Screenshot 2025-05-25 at 23.13.12.png)

## 数据增强

训练过程中使用以下数据增强方法：
- 随机水平翻转
- 随机垂直翻转
- 随机旋转
- 随机仿射变换
- 颜色抖动
- 标准化

## 评估指标

模型评估包含以下指标：
- 准确率 (Accuracy)
- F1 分数
- 精确率 (Precision)
- 召回率 (Recall)
- 混淆矩阵
- ROC 曲线和 AUC 值

## 训练过程（50 Epochs）

![Screenshot 2025-05-25 at 23.16.44](../../Screenshot 2025-05-26 at 07.00.14.png)

## 结果

