import cv2
import logging
from ultralytics import YOLO
import torch

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def detect(self, image, confidence_threshold=0.5):
        """
        检测给定图片中的目标，并返回带有检测框的图片。

        参数:
            image (numpy.ndarray): 图片数据
            confidence_threshold (float): 置信度阈值，默认为0.5

        返回:
            numpy.ndarray: 带有检测框的图片
        """
        image_tensor = self.preprocess(image).to(self.device)
        results = self.model(image_tensor)
        detected_image = self.postprocess(image, results, confidence_threshold)
        return detected_image

    def detect_with_info(self, image, confidence_threshold=0.3):
        """
        检测给定图片中的目标，并返回带有检测框的图片以及检测信息。

        参数:
            image (numpy.ndarray): 图片数据
            confidence_threshold (float): 置信度阈值，默认为0.5

        返回:
            tuple: (numpy.ndarray, list) 带有检测框的图片和检测信息列表
        """
        image_tensor = self.preprocess(image).to(self.device)
        results = self.model(image_tensor)
        detected_image, detection_info = self.postprocess_with_info(image, results, confidence_threshold)
        return detected_image, detection_info

    def preprocess(self, image):
        """
        预处理图像，转换为模型输入格式

        参数:
            image (numpy.ndarray): 图片数据

        返回:
            torch.Tensor: 预处理后的图像张量
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image / 255.0  # 归一化
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image

    def postprocess(self, image, results, confidence_threshold):
        """
        后处理模型输出，绘制检测框

        参数:
            image (numpy.ndarray): 原始图片数据
            results (list): 模型输出结果
            confidence_threshold (float): 置信度阈值

        返回:
            numpy.ndarray: 带有检测框的图片
        """
        for result in results:
            boxes = result.boxes.cpu()  # 将检测框从GPU移回CPU
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # 获取类别ID
                confidence = float(box.conf[0])  # 获取置信度
                if class_id == 0 and confidence >= confidence_threshold:  # 假设类别0表示行人
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {confidence:.2f}"  # 添加置信度
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def postprocess_with_info(self, image, results, confidence_threshold):
        """
        后处理模型输出，绘制检测框并返回检测信息

        参数:
            image (numpy.ndarray): 原始图片数据
            results (list): 模型输出结果
            confidence_threshold (float): 置信度阈值

        返回:
            tuple: (numpy.ndarray, list) 带有检测框的图片和检测信息列表
        """
        detection_info = []
        for result in results:
            boxes = result.boxes.cpu()  # 将检测框从GPU移回CPU
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
