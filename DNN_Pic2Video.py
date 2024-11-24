import cv2
import os
import numpy as np

class DNN_PeopleIdentify:
    def __init__(self, model_path, config_path, class_names_path):
        """
        初始化 DNN_PeopleIdentify 类

        :param model_path: YOLO 模型文件路径
        :param config_path: YOLO 配置文件路径
        :param class_names_path: 类别名称文件路径
        """
        self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        self.class_names = self._load_class_names(class_names_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def _load_class_names(self, class_names_path):
        with open(class_names_path, 'rt') as f:
            return f.read().rstrip('\n').split('\n')

    def detect_people(self, image):
        """
        检测图像中的行人

        :param image: 输入图像
        :return: 行人框的坐标列表 [(x, y, w, h)]
        """
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # 0 表示 "person" 类别
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    width = int(detection[2] * image.shape[1])
                    height = int(detection[3] * image.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        people_boxes = [boxes[i] for i in indices]

        return people_boxes
