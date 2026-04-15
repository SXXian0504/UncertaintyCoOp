# UncertaintyCoOp: # Uncertainty-Aware Prompt Learning for Multi-Label Recognition with Incomplete Annotations

[English](README.md) | [中文](README_CN.md)

文章已录用。

## 简介

多标签图像识别通常假设标注是完整的，然而真实世界的数据集（如MS-COCO和PASCAL VOC）往往包含缺失或错误的标签。将未标注的类别视为负样本会引入假负样本监督，导致在部分标注条件下性能下降。为了解决这个问题，我们提出了不确定性引导上下文优化（Uncertainty-Guided Context Optimization, UncertaintyCoOp），这是一个显式建模预测不确定性的提示学习框架，用于部分标签多标签识别。

UncertaintyCoOp包含三个组件：(1) 熵-置信度混合不确定性估计器，用于捕捉认知和偶然不确定性；(2) 不确定性引导的提示融合机制，结合正、负和不确定提示；(3) 具有动量更新教师的不确定性感知损失，以确保在噪声监督下进行稳定优化。

## 动机

![动机示意图](assets/Motivation.png)

VOC和COCO数据集中错误和缺失标注的示意图。每个示例(a-d)比较了数据集的真实标注(GT)与实际图像内容(Actual)。$\checkmark$和$\times$分别表示存在和不存在的标签。错误和缺失的标注分别用深蓝色和灰色虚线框突出显示。

在实际的多标签图像识别任务中，数据集的标注往往是不完整和不准确的。这种标注噪声会严重影响模型的性能，因为传统的学习方法会将缺失的标签错误地视为负样本，从而导致模型学习到错误的知识。我们的方法通过显式建模预测不确定性来解决这个问题。

## 框架

![框架示意图](assets/framework.png)

UncertaintyCoOp框架用于部分标签多标签识别的示意图。对于每个类别，正提示、不确定提示和可学习的负嵌入与图像块特征交互，产生三个方向的预测。熵-置信度不确定性估计器生成不确定性系数，该系数自适应地融合多分支预测，并进一步指导具有可靠性教师分布的不确定性感知损失函数。

## 环境配置

### 1. 创建conda环境

```bash
conda env create -f environment.yaml
conda activate uncertaintycoop
```

### 2. 安装Dassl

具体按照 https://github.com/KaiyangZhou/Dassl.pytorch 配置Dassl。


### 3. 检验CUDA和Dassl的可用性

#### 检查 Dassl
```bash
python -c "import importlib.util; print('Dassl installed:', importlib.util.find_spec('dassl') is not None)"
```

#### 检查 CUDA
```
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

## 数据集

本项目使用VOC2007和MSCOCO数据集，请前往官网下载。

### VOC2007数据集文件结构

```
VOC2007/
    ├── Annotations/          # 标注（XML，每张图一个）
    │   ├── 000001.xml
    │   ├── 000002.xml
    │   └── ...
    │
    ├── JPEGImages/          # 原始图片
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    │
    ├── ImageSets/
    │   └── Main/            # 划分文件（关键！）
    │       ├── train.txt
    │       ├── val.txt
    │       ├── trainval.txt
    │       ├── test.txt
    │       │
    │       ├── aeroplane_train.txt
    │       ├── aeroplane_val.txt
    │       └── ...（每个类别一个）
```

### MSCOCO数据集文件结构

```
coco/
├── annotations/
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
│
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
│
├── val2014/
│   ├── COCO_val2014_000000000139.jpg
│   ├── COCO_val2014_000000000285.jpg
│   └── ...
```

## 训练

### VOC2007

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/voc2007.yaml --train_batch_size 32 --input_size 448 --lr 8e-2 --max_epochs 50 --loss_w 0.03 -pp 0.9 --csc --method_name uncertaintycoop --warmup_epochs 1
```

### MSCOCO

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/coco.yaml --train_batch_size 32 --input_size 448 --lr 8e-2 --max_epochs 50 --loss_w 0.03 -pp 0.9 --csc --method_name uncertaintycoop --warmup_epochs 1
```

### 参数说明

| 参数名 | 含义 | 示例 | 备注 |
|--------|------|------|------|
| config_file | 模型配置文件 | configs/models/rn101_ep50.yaml | 定义模型架构（如ResNet101）、训练轮数、优化器等 |
| datadir | 数据集路径 | ../datasets/mscoco_2014/ | 指定本地数据集目录 |
| dataset_config_file | 数据集配置文件 | configs/datasets/coco.yaml 或 voc2007.yaml | 告诉程序数据集格式、类别数等信息 |
| input_size | 输入图像大小 | 448 | 图像会被resize到448×448 |
| lr | 学习率 | 8e-2| 控制训练更新步幅 |
| loss_w | loss权重系数 | 0.03 | 用来平衡不同缺失比例下的损失幅度 |
| pp | 可用标签比例 | 0到1之间| 0.5表示只有50%的标签是已知的（partial label率） |
| --csc | 类特定提示 | 无需值（布尔开关） | pp较小时使用，是否启用"类特定提示词"，若不加此参数则使用class-agnostic prompt |

## 验证

### VOC2007

```bash
CUDA_VISIBLE_DEVICES=0 python val.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/VOC2007.yaml --input_size 448 --pretrained <your_model_path> --csc --method_name uncertaintycoop
```

### MSCOCO

```bash
CUDA_VISIBLE_DEVICES=0 python val.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/COCO.yaml --input_size 448 --pretrained <your_model_path> --csc --method_name uncertaintycoop
```
