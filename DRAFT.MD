要求：

1. 用中文版写出文章的简介：
可以参考论文摘要（Multi-label image recognition typically assumes exhaustive annotations, whereas real-world datasets such as MS-COCO and PASCAL VOC often contain missing or incorrect labels. Treating unlabeled categories as negatives introduces false-negative supervision, leading to degraded performance under partial annotations. 
To address this issue, we propose Uncertainty-Guided Context Optimization (UncertaintyCoOp), a prompt learning framework that explicitly models prediction uncertainty for partial-label multi-label recognition.
UncertaintyCoOp consists of three components: (1) an entropy-confidence hybrid uncertainty estimator capturing epistemic and aleatoric uncertainty, (2) an uncertainty-guided prompt fusion mechanism combining positive, negative, and uncertain prompts, and (3) an uncertainty-aware loss with a momentum-updated teacher to ensure stable optimization under noisy supervision.）

2. 用中文写出文章的motivation，并插入 @assets/Motivation.png  的配图。文本部分按照论文简单解释即可。 
可以参考（Illustration of incorrect and missing annotations in VOC and COCO datasets. Each example (a-d) compares the dataset ground truth (GT) with the actual image content (Actual). $\checkmark$ and $\times$ denote present and absent labels, respectively. Incorrect and missing annotations are highlighted in dark blue and gray dashed boxes, respectively.）

3. 用中文写出framework，并插入 @assets/framework.png  的配图。 文本部分按照论文简单解释即可。 
可以参考（Illustration of the proposed UncertaintyCoOp framework for partial-label MLR.
For each category, a positive prompt, an uncertainty prompt, and a learnable negative embedding interact with image patch features to produce three directional predictions. The entropy–confidence uncertainty estimator generates an uncertainty coefficient, which adaptively fuses the multi-branch predictions and further guides the uncertainty-aware loss function with a reliability teacher distribution.）

3. 用中文写出环境配置文档。环境所需依赖可以按照environment.yaml配置，并且按照https://github.com/KaiyangZhou/Dassl.pytorch 配置Dassl。

（1）创建conda环境
（2）安装Dassl
（3）检验cuda、Dassl的可用性
# 检查 Dassl
python -c "import importlib.util; print('Dassl installed:', importlib.util.find_spec('dassl') is not None)"

# 检查 CUDA
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"



4. 数据集部分：
使用的是VOC2007和MSCOCO，可以到官网下载。根据我的文件夹情况，给出数据集文件树。
VOC：
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

MSCOCO:
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

5.训练：

VOC：
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/voc2007.yaml --train_batch_size 32 --input_size 448 --lr 8e-2 --max_epochs 50 --loss_w 0.03 -pp 0.9 --csc --method_name uncertaintycoop --warmup_epochs 1

MSCOCO：
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/coco.yaml --train_batch_size 32 --input_size 448 --lr 8e-2 --max_epochs 50 --loss_w 0.03 -pp 0.9 --csc --method_name uncertaintycoop --warmup_epochs 1 

参数名	含义	示例	备注
config_file	模型配置文件	configs/models/rn101_ep50.yaml	定义模型架构（如 ResNet101）、训练轮数、优化器等
datadir	数据集路径	../datasets/mscoco_2014/	指定本地数据集目录
dataset_config_file	数据集配置文件	configs/datasets/coco.yaml 或 voc2007.yaml	告诉程序数据集格式、类别数等信息
input_size	输入图像大小	448	图像会被 resize 到 448×448
lr	学习率	0.001 (VOC) 或 0.002 (COCO)	控制训练更新步幅
loss_w	loss 权重系数	0.03	用来平衡不同缺失比例下的损失幅度
pp	portion of available labels	0到1之间，0.5	表示只有 50% 的标签是已知的（partial label 率）
--csc	class-specific prompts	无需值（布尔开关）
pp较小时使用	是否启用“类特定提示词”，若不加此参数则使用 class-agnostic prompt


6.验证：

VOC：

CUDA_VISIBLE_DEVICES=0 python val.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/VOC2007.yaml --input_size 448 --pretrained <your_model_path> --csc --method_name uncertaintycoop

MSCOCO：

CUDA_VISIBLE_DEVICES=0 python val.py --config_file configs/models/rn101_adam.yaml --datadir <your_dataset_path> --dataset_config_file configs/datasets/COCO.yaml --input_size 448 --pretrained <your_model_path> --csc --method_name uncertaintycoop