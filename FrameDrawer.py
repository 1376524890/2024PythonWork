# FrameDrawer.py
import cv2

class FrameDrawer:
    def __init__(self, text="Sample Text", font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 0, 0), thickness=2):
        """
        初始化 FrameDrawer 类

        :param text: 要绘制的文字
        :param font: 字体类型
        :param font_scale: 字体大小
        :param font_color: 字体颜色
        :param thickness: 字体厚度
        """
        self.text = text
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness

    def draw_faces(self, frame, faces):
        """
        在图像上绘制人脸方框

        :param frame: 图像数据
        :param faces: 人脸坐标列表
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_position = (x - 10, y - 10)
            cv2.putText(frame, self.text, text_position, self.font, self.font_scale, self.font_color, self.thickness)
