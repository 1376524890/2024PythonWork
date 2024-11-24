# peopleIdentify.py
import cv2


def load_face_cascade():
    """
    加载预训练的人体检测模型

    返回:
        cv2.CascadeClassifier: 人体检测模型
    """
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


def detect_faces(image, face_cascade):
    """
    检测给定图片中的人体，并返回人体的坐标。

    参数:
        image (numpy.ndarray): 图片数据
        face_cascade (cv2.CascadeClassifier): 人体检测模型

    返回:
        list: 人体坐标的列表，每个元素为 (x, y, w, h)，其中 x 和 y 是人体框左上角的坐标，w 和 h 分别是宽度和高度。
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人体
    bodies = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    """
    # 输出每个人体的位置
    for (x, y, w, h) in bodies:
        print(f"检测到人体位置：({x}, {y}), 宽度：{w}, 高度：{h}")
    """
    return bodies

'''
# 使用示例
image_path = 'frames/frame_0000.jpg'  # 替换为你的图片路径
face_cascade = load_face_cascade()
image = cv2.imread(image_path)
bodies = detect_faces(image, face_cascade)
print(bodies)
'''
