import cv2
import numpy as np

class DNN_PeopleIdentify:
    def __init__(self, model_path, config_path):
        """
        初始化 DNN 模型

        参数:
            model_path (str): 模型文件路径
            config_path (str): 配置文件路径
        """
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    def detect_people(self, image):
        """
        检测给定图片中的行人，并返回行人的坐标。

        参数:
            image (numpy.ndarray): 图片数据

        返回:
            list: 行人坐标的列表，每个元素为 (x, y, w, h)，其中 x 和 y 是行人框左上角的坐标，w 和 h 分别是宽度和高度。
        """
        # 获取图像的宽度和高度
        (h, w) = image.shape[:2]

        # 构建输入 blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # 将 blob 输入网络
        self.net.setInput(blob)
        detections = self.net.forward()

        # 初始化行人列表
        people = []

        # 遍历检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 过滤掉低置信度的检测结果
            if confidence > 0.5:
                # 获取行人框的坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 添加到行人列表
                people.append((startX, startY, endX - startX, endY - startY))

        return people

"""
# 使用示例
model_path = 'path_to_model.caffemodel'  # 替换为你的模型文件路径
config_path = 'path_to_config.prototxt'  # 替换为你的配置文件路径
image_path = 'frames/frame_0000.jpg'  # 替换为你的图片路径

people_identify = DNN_PeopleIdentify(model_path, config_path)
image = cv2.imread(image_path)
people = people_identify.detect_people(image)
print(people)
"""