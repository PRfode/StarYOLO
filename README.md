# StarYOLO

基于 YOLOv8 的改进项目，使用 StarNet 模块和 MambaYOLO 架构替换 YOLOv8 backbone，
结合冻结预训练 + 训练新模块的两阶段策略。

所有改动基本围绕 Ultralytics 库进行，这个库还是太香了

## 目录结构

```
StarYOLO/
├── ultralytics/              # MambaYOLO 项目源码的修改版
│   ├── cfg/models/           # 所有模型架构 YAML 配置
│   │   ├── yolov8n.yaml      # 标准 YOLOv8n
│   │   ├── yolov8nmod.yaml   # layer 8 C2f 替换为 StarNetBlock
│   │   └── staryolon.yaml    # 全 StarNet backbone + SimpleStem
│   ├── nn/modules/
│   │   ├── star_net.py       # StarBlock / StarNetBlock / SimpleStem / VisionClueMerge
│   │   └── ...
│   └── engine/
│       ├── model.py          # YOLO 主类（含 CKPT 陷阱）
│       └── trainer.py        # 训练器（含 freeze 输出）
├── res/
│   ├── dataset/              # 数据集文件夹
│   │   ├── coco512/          # COCO2017 子集 (507 train + 128 val, 80类)
│   │   │   ├── images/
│   │   │   │   ├── train2017/
│   │   │   │   └── val2017/
│   │   │   └── labels/
│   │   │       ├── train2017/
│   │   │       └── val2017/
│   │   ├── ...
│   │   └── dataset_cfg/      # 所有数据集 YAML 配置
│   │       └── coco512.yaml  # path + train/val 路径 + 类别名
│   ├── model/                # 预训练 / 训练好的 .pt 权重
│   │   ├── ...
│   │   └── yolov8n.pt        # 官方 YOLOv8n 预训练权重
│   └── mapping/              # 层映射表 (JSON)
│       ├── yolov8n_staryolon.json
│       └── ...
├── script/
│   ├── train.py               # 旧版：加载 + 跳过指定层 + 训练
│   ├── train_with_frozen.py   # [推荐] 基于映射文件的训练脚本
│   ├── train_yolov8n_backbone.py  # 仅训练 backbone 实验
│   └── predict.py             # 验证集评估 (mAP50 / mAP75 / mAP)
└── runs/                      # 训练中间结果 / 日志 / 权重

```

注意以下几点：
- 数据集数据和配置文件都放在 `res/dataset/` 目录下，但是模型权重放在 `res/model/` 目录下，配置文件放在 `ultralytics/cfg/models/`


## 数据集：coco512

COCO2017 的子集，轻量级验证用：

| 子集 | 图片数 | 备注 |
|---|---|---|
| train2017 | 507 张 + 5 背景 | 从 COCO2017 train 取前 512 张 |
| val2017 | 128 张 | 从 COCO2017 val 取前 128 张 |

配置文件位于 `res/dataset/dataset_cfg/coco512.yaml`，包含 `path`、`train`/`val` 路径和 80 个 COCO 类别名。

## 模型架构

| YAML 文件 | 层数 | 说明 |
|---|---|---|
| `yolov8n.yaml` | 23 | 标准 YOLOv8n，backbone: Conv + C2f + SPPF |
| `yolov8nmod.yaml` | 23 | backbone layer 8 替换为 StarNetBlock，其余同 yolov8n |
| `staryolon.yaml` | 22 | 全新 backbone: SimpleStem + StarNetBlock + VisionClueMerge，head 同 yolov8 |

所有模型的 `width_multiple=0.25`，`depth_multiple=0.33`（nano 规格）。

## 层映射系统

用于在不同架构之间复用预训练权重。映射表定义哪些层从预训练加载、哪些层随机初始化。

### 映射文件格式

`res/mapping/{预训练模型名}_{目标模型名}.json`:

```json
{
    "pretrained_layer_idx": "target_layer_idx",
    ...
}
```

未被映射的层会被随机初始化。

只要也应该为

### 现有映射

| 映射文件 | 策略 |
|---|---|
| `yolov8n_yolov8nmod.json` | 全等映射，丢弃 layer 8（C2f → StarNetBlock 参数不兼容） |
| `yolov8n_staryolon.json` | 只保留 head (10-22 → 9-21)，backbone (0-9) 全部丢弃 |

### 训练流程

1. 加载目标模型 YAML → 随机初始化
2. 读取映射表 → 从预训练权重复制映射层参数
3. 未映射层保持随机初始化
4. **冻结已映射层**（有预训练权重）、**训练未映射层**（随机初始化）

### 训练命令

```bash
# 推荐：基于映射文件
python script/train_with_frozen.py \
    --pt yolov8n --model yolov8nmod --dataset coco512 --epochs 50
```

训练的更多参数查看 `script/train_with_frozen.py` 的 argparse。

## 评估

```bash
python script/predict.py \
    --model yolov8n --dataset coco512
```

## 环境

Python 3.10 + Torch 2.5.1

剩下缺什么就直接pip install即可

**不要 `pip install ultralytics`** 使用项目的修改版 `ultralytics/`(修改自MambaYOLO)


注意事项：
  - DataLoader 建议 `--workers 0` 自动分配（也许？
  - Unicode 输出建议 `PYTHONIOENCODING=utf-8`

## 参考项目

- [YOLOv8](https://github.com/haermosi/yolov8)
- [MambaYOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [StarNet](https://github.com/ma-xu/Rewrite-the-Stars)
