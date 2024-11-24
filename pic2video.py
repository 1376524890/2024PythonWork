# pic2video.py
import cv2
import os
import glob

import peopleIdentify
from FrameDrawer import FrameDrawer


class ImageSequencePlayer:
    def __init__(self, image_dir, frame_rate=30):
        """
        初始化 ImageSequencePlayer 类

        :param image_dir: 图片文件所在的目录
        :param frame_rate: 播放帧率，默认为30帧/秒
        """
        self.image_dir = image_dir
        self.frame_rate = frame_rate
        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.face_cascade = peopleIdentify.load_face_cascade()  # 加载人脸检测模型
        self.frame_drawer = FrameDrawer()  # 创建 FrameDrawer 实例

    def play(self):
        """
        按指定帧率播放图片序列
        """
        delay = int(1000 / self.frame_rate)  # 计算每帧之间的延迟时间（毫秒）

        for image_path in self.images:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"无法读取图片: {image_path}")
                continue

            # 检测人脸
            faces = peopleIdentify.detect_faces(frame, self.face_cascade)

            # 绘制方框和文字
            self.frame_drawer.draw_faces(frame, faces)

            cv2.imshow('Image Sequence', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    image_dir = 'img1'
    player = ImageSequencePlayer(image_dir, frame_rate=30)
    player.play()
