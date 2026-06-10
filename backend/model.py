import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO

class MyYOLO:
    def __init__(self, args):
        self.ROOT = os.path.abspath('.') + "/"

        self.WEIGHTS = self.ROOT + args.weights
        self.MODEL_YAML = self.ROOT + args.model_yaml

        self.model = YOLO(self.MODEL_YAML)
        self.model.load(self.WEIGHTS)  # 加载训练好的权重

    def predict(self, image):
        """
        image_path: str, 图片路径
        visualize: bool, 是否显示可视化结果
        """
        # 预测
        results = self.model.predict(source=image, imgsz=640, conf=0.25, verbose=False)
        result = results[0]

        img = image.copy()
        img = np.array(img)  # PIL -> numpy
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = f"{result.names[int(cls)]}: {conf:.2f}"
            color = (255, 0, 0)  # 蓝色框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img