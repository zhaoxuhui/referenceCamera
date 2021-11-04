# coding=utf-8
import cv2
import time


# 按钮回调事件
def reactionEvent(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # shot按钮
        if start_x < x < start_x + 70 and start_y < y < start_y + 30:
            cv2.imwrite(time.strftime("shot-%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", frame)
            print("saved as:" + time.strftime("shot-%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
        # reference按钮
        elif start_x + 80 < x < start_x + 225 and start_y < y < start_y + 30:
            cv2.imwrite("reference.jpg", frame)
            print("take as reference")
            global ref_frame, flag_ref
            ref_frame = cv2.imread("reference.jpg")
            flag_ref = 1
        # exit按钮
        elif start_x + 235 < x < start_x + 300 and start_y < y < start_y + 30:
            cv2.destroyAllWindows()
            exit()


def nothing(x):
    pass


global ref_frame

# 脚本实现拍摄某张影像作为参考，再拍摄其它照片的功能
if __name__ == '__main__':
    flag_ref = 0
    start_x = 30
    start_y = 30
    # 新建一个VideoCapture对象，指定第0个相机进行视频捕获
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("img_cam")
    cv2.setMouseCallback("img_cam", reactionEvent)
    cv2.createTrackbar('alpha', "img_cam", 50, 100, nothing)

    # 一直循环捕获，直到手动退出
    while 1:
        # 返回两个值，ret表示读取是否成功，frame为读取的帧内容
        ret, frame = cap.read()

        # 判断传入的帧是否为空，为空则退出
        if frame is None:
            break
        else:
            if flag_ref != 1:
                # 如果没有参考影像，就拷贝当前帧影像
                frame_canvas = frame.copy()
            elif flag_ref == 1:
                # 如果有参考影像，就根据滑动条的位置获取比例，对当前帧和参考帧进行混合并输出
                alpha_value = cv2.getTrackbarPos('alpha', "img_cam")
                alpha_value = alpha_value / 100
                img_mix = cv2.addWeighted(ref_frame, alpha_value, frame, 1 - alpha_value, 0)
                frame_canvas = img_mix.copy()

            # 按钮绘制
            cv2.rectangle(frame_canvas, (start_x, start_y), (start_x + 70, start_y + 30), (0, 0, 255), 2)
            cv2.putText(frame_canvas, "shot", (start_x, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        1, cv2.LINE_AA)
            cv2.rectangle(frame_canvas, (start_x + 80, start_y), (start_x + 225, start_y + 30), (0, 0, 255), 2)
            cv2.putText(frame_canvas, "reference", (start_x + 80, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame_canvas, (start_x + 235, start_y), (start_x + 300, start_y + 30), (0, 0, 255), 2)
            cv2.putText(frame_canvas, "exit", (start_x + 240, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # 调用OpenCV图像显示函数显示每一帧
            cv2.imshow("img_cam", frame_canvas)
            cv2.waitKey(1)

    # 释放VideoCapture对象
    cap.release()
