import cv2
import logging
from ultralytics import YOLO

# 关闭日志输出
logging.getLogger('ultralytics').setLevel(logging.WARNING)

class YOLOv8Detector:
    def __init__(self, model_path):
        """
        初始化 YOLOv8 模型

        参数:
            model_path (str): 模型文件路径
        """
        self.model = YOLO(model_path)

    def detect(self, image, confidence_threshold=0.5):
        """
        检测给定图片中的目标，并返回带有检测框的图片。

        参数:
            image (numpy.ndarray): 图片数据
            confidence_threshold (float): 置信度阈值，默认为0.5

        返回:
            numpy.ndarray: 带有检测框的图片
        """
        results = self.model(image)
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # 获取类别ID
                confidence = float(box.conf[0])  # 获取置信度
                if class_id == 0 and confidence >= confidence_threshold:  # 假设类别0表示行人
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {confidence:.2f}"  # 添加置信度
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def detect_with_info(self, image, confidence_threshold=0.3):
        """
        检测给定图片中的目标，并返回带有检测框的图片以及检测信息。

        参数:
            image (numpy.ndarray): 图片数据
            confidence_threshold (float): 置信度阈值，默认为0.5

        返回:
            tuple: (numpy.ndarray, list) 带有检测框的图片和检测信息列表
        """
        results = self.model(image)
        detection_info = []
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # 获取类别ID
                confidence = float(box.conf[0])  # 获取置信度
                if class_id == 0 and confidence >= confidence_threshold:  # 假设类别0表示行人
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {confidence:.2f}"  # 添加置信度
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
                    detection_info.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        return image, detection_info
