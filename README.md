# StarYOLO
该项目源自YOLOv8，将使用YOLOv8为预训练模型对Backbone部分进行修改。

该架构借鉴MambaYOLO的架构，使用轻量化的StarNet替换。经过冻结训练和联合训练双阶段得到最终模型。


## 注意：
### 数据集
将数据集放在res/dataset下，形成类似于 `./res/dataset/coco128/images/train2017/` 的目录结构

详情查看 `res/dataset/dataset_cfg/*.yaml` 配置文件的结构

### 环境
使用python3.10，torch2.5.1，torchaudio2.5.1，torchvision0.20.1

剩下缺什么直接pip install

### 运行
脚本执行位置在根目录，而不是在scripts文件夹下。

## reference
- [YOLOv8](https://github.com/haermosi/yolov8)
- [MambaYOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [StarNet](https://github.com/ma-xu/Rewrite-the-Stars)