# video2pic.py
import cv2
import os

class VideoToImages:
    def __init__(self, video_path, output_dir):
        """
        初始化 VideoToImages 类

        :param video_path: 视频文件的路径
        :param output_dir: 保存图片的目录
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(self.video_path)

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def convert(self, frame_rate=1):
        """
        将视频转换为图片序列

        :param frame_rate: 每秒保存的帧数，默认为1帧/秒
        """
        frame_count = 0
        success, frame = self.cap.read()
        while success:
            # 计算当前帧的时间戳
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # 检查是否需要保存当前帧
            if current_time >= frame_count / frame_rate:
                # 缩小帧的分辨率
                resized_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
                image_path = os.path.join(self.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(image_path, resized_frame)
                frame_count += 1

            success, frame = self.cap.read()

        # 释放视频捕获对象
        self.cap.release()
        print(f"转换完成，共保存 {frame_count} 帧图片到 {self.output_dir}")

# 示例用法
if __name__ == "__main__":
    video_path = 'D:/pythonProject/2024PythonWork/your_video.mp4'
    output_dir = 'D:/pythonProject/2024PythonWork/frames'

    converter = VideoToImages(video_path, output_dir)
    converter.convert(frame_rate=30)
