import cv2
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(self, model_path):
        """
        初始化 YOLOv8 模型

        参数:
            model_path (str): 模型文件路径
        """
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        检测给定图片中的目标，并返回带有检测框的图片。

        参数:
            image (numpy.ndarray): 图片数据

        返回:
            numpy.ndarray: 带有检测框的图片
        """
        results = self.model(image)
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # 获取类别ID
                if class_id == 0:  # 假设类别0表示行人
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {box.conf[0]:.2f}"  # 添加置信度
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def detect_with_info(self, image):
        """
        检测给定图片中的目标，并返回带有检测框的图片以及检测信息。

        参数:
            image (numpy.ndarray): 图片数据

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
                if class_id == 0:  # 假设类别0表示行人
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {box.conf[0]:.2f}"  # 添加置信度
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detection_info.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf[0])
                    })
        return image, detection_info