import cv2
import os

from DNN_Pic2Video import DNN_PeopleIdentify


class VideoToImages:
    def __init__(self, video_path, output_dir, model_path, config_path, class_names_path):
        """
        初始化 VideoToImages 类

        :param video_path: 视频文件的路径
        :param output_dir: 保存图片的目录
        :param model_path: YOLO 模型文件路径
        :param config_path: YOLO 配置文件路径
        :param class_names_path: 类别名称文件路径
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(self.video_path)

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 初始化 DNN_PeopleIdentify 类
        self.people_identify = DNN_PeopleIdentify(model_path, config_path, class_names_path)

    def convert(self, frame_rate=1):
        """
        将视频转换为图片序列，并进行行人识别

        :param frame_rate: 每秒保存的帧数，默认为1帧/秒
        """
        frames = self.video_to_images(frame_rate)
        for frame_count, frame in frames:
            # 缩小帧的分辨率
            resized_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            image_path = os.path.join(self.output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(image_path, resized_frame)

            # 进行人识别
            people = self.people_identify.detect_people(resized_frame)

            # 在图像上绘制行人框
            for (x, y, w, h) in people:
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 保存带有行人框的图像
            annotated_image_path = os.path.join(self.output_dir, f"annotated_frame_{frame_count:04d}.jpg")
            cv2.imwrite(annotated_image_path, resized_frame)

        # 释放视频捕获对象
        self.cap.release()
        print(f"转换完成，共保存 {len(frames)} 帧图片到 {self.output_dir}")

    def video_to_images(self, frame_rate=1):
        """
        将视频转换为图片序列

        :param frame_rate: 每秒保存的帧数，默认为1帧/秒
        :return: 生成器，生成 (frame_count, frame) 对
        """
        frame_count = 0
        success, frame = self.cap.read()
        while success:
            # 计算当前帧的时间戳
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # 检查是否需要保存当前帧
            if current_time >= frame_count / frame_rate:
                yield frame_count, frame
                frame_count += 1

            success, frame = self.cap.read()

# 示例用法
if __name__ == "__main__":
    video_path = 'D:/pythonProject/2024PythonWork/your_video.mp4'
    output_dir = 'D:/pythonProject/2024PythonWork/frames'
    model_path = 'path_to_yolov3.weights'  # 替换为你的模型文件路径
    config_path = 'path_to_yolov3.cfg'  # 替换为你的配置文件路径
    class_names_path = 'path_to_coco.names'  # 替换为你的类别名称文件路径

    converter = VideoToImages(video_path, output_dir, model_path, config_path, class_names_path)
    converter.convert(frame_rate=30)
