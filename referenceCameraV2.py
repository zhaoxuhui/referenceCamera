# coding=utf-8
import cv2
import time
import math


def reactionEvent(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_x < x < start_x + 70 and start_y < y < start_y + 30:
            cv2.imwrite(time.strftime("shot-%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", frame)
            print("saved as:" + time.strftime("shot-%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
        elif start_x + 80 < x < start_x + 225 and start_y < y < start_y + 30:
            cv2.imwrite("reference.jpg", frame)
            print("take as reference")
            global ref_frame, flag_ref
            ref_frame = cv2.imread("reference.jpg")
            global ref_kps, ref_des
            ref_kps, ref_des = getOrbKps(ref_frame)
            flag_ref = 1
        elif start_x + 235 < x < start_x + 300 and start_y < y < start_y + 30:
            cv2.destroyAllWindows()
            exit()


def nothing(x):
    pass


def getOrbKps(img, numKps=2000):
    """
    获取ORB特征点和描述子
    :param img: 读取的输入影像
    :param numKps: 期望提取的特征点个数，默认2000
    :return: 特征点和对应的描述子
    """

    orb = cv2.ORB_create(nfeatures=numKps)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def cvtCvKeypointToNormal(keypoints):
    """
    将OpenCV中KeyPoint类型的特征点转换成(x,y)格式的普通数值
    :param keypoints: KeyPoint类型的特征点列表
    :return: 转换后的普通特征点列表
    """

    cvt_kps = []
    for i in range(keypoints.__len__()):
        cvt_kps.append((keypoints[i].pt[0], keypoints[i].pt[1]))
    return cvt_kps


def bfMatch(kp1, des1, kp2, des2, disTh=15.0):
    """
    基于BF算法的匹配
    :param kp1: 特征点列表1
    :param des1: 特征点描述列表1
    :param kp2: 特征点列表2
    :param des2: 特征点描述列表2
    :return: 匹配的特征点对
    """

    good_kps1 = []
    good_kps2 = []
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    if matches.__len__() == 0:
        return good_kps1, good_kps2
    else:
        min_dis = 10000
        for item in matches:
            dis = item.distance
            if dis < min_dis:
                min_dis = dis

        g_matches = []
        for match in matches:
            if match.distance <= max(1.1 * min_dis, disTh):
                g_matches.append(match)

        # print("matches:" + g_matches.__len__().__str__())

        cvt_kp1 = []
        cvt_kp2 = []
        if type(kp1[0]) is cv2.KeyPoint:
            cvt_kp1 = cvtCvKeypointToNormal(kp1)
        else:
            cvt_kp1 = kp1
        if type(kp2[0]) is cv2.KeyPoint:
            cvt_kp2 = cvtCvKeypointToNormal(kp2)
        else:
            cvt_kp2 = kp2

        for i in range(g_matches.__len__()):
            good_kps1.append([cvt_kp1[g_matches[i].queryIdx][0], cvt_kp1[g_matches[i].queryIdx][1]])
            good_kps2.append([cvt_kp2[g_matches[i].trainIdx][0], cvt_kp2[g_matches[i].trainIdx][1]])

        return good_kps1, good_kps2


global ref_frame
global ref_kps
global ref_des

# 脚本实现拍摄某张影像作为参考，再拍摄其它照片的功能
# 相比于V1，增加了运动提示显示
if __name__ == '__main__':
    flag_ref = 0
    start_x = 30
    start_y = 30
    # 新建一个VideoCapture对象，指定第0个相机进行视频捕获
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("img_cam")
    cv2.setMouseCallback("img_cam", reactionEvent)
    cv2.createTrackbar('alpha', 'img_cam', 50, 100, nothing)

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
                # 提取当前帧ORB特征并和参考帧ORB匹配
                frame_kps, frame_des = getOrbKps(frame)
                g_kp1, g_kp2 = bfMatch(ref_kps, ref_des, frame_kps, frame_des)

                # 如果有参考影像，就根据滑动条的位置获取比例，对当前帧和参考帧进行混合并输出
                alpha_value = cv2.getTrackbarPos('alpha', "img_cam")
                alpha_value = alpha_value / 100
                img_mix = cv2.addWeighted(ref_frame, alpha_value, frame, 1 - alpha_value, 0)
                frame_canvas = img_mix.copy()
                frame_width = frame_canvas.shape[1]
                frame_height = frame_canvas.shape[0]

                # 计算参考帧和当前帧之间的位置差异
                diff_xs = []
                diff_ys = []
                for i in range(len(g_kp1)):
                    kp1_x = int(g_kp1[i][0])
                    kp1_y = int(g_kp1[i][1])
                    kp1 = (kp1_x, kp1_y)
                    kp2_x = int(g_kp2[i][0])
                    kp2_y = int(g_kp2[i][1])
                    diff_xs.append(kp2_x - kp1_x)
                    diff_ys.append(kp2_y - kp1_y)

                diff_xs.sort()
                diff_ys.sort()
                median_diff_x = diff_xs[int((len(diff_xs) - 1) / 2)]
                median_diff_y = diff_ys[int((len(diff_ys) - 1) / 2)]
                diff_d = math.sqrt(median_diff_x ** 2 + median_diff_y ** 2)

                # 根据不同的距离设置不同颜色
                color = (0, 0, 255)
                if diff_d < 5:
                    color = (0, 255, 0)
                elif diff_d > 5 and diff_d < 20:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                # 绘制匹配点对
                for i in range(len(g_kp1)):
                    kp1_x = int(g_kp1[i][0])
                    kp1_y = int(g_kp1[i][1])
                    kp1 = (kp1_x, kp1_y)
                    kp2_x = int(g_kp2[i][0])
                    kp2_y = int(g_kp2[i][1])
                    kp2 = (kp2_x, kp2_y)
                    cv2.circle(frame_canvas, kp1, 1, color)
                    cv2.circle(frame_canvas, kp2, 1, color)
                    cv2.line(frame_canvas, kp1, kp2, color, 1)

                # 根据不同差异绘制运动提示箭头
                if median_diff_y > 5:
                    # 向上
                    cv2.arrowedLine(frame_canvas, (int(frame_width / 2), int(frame_height / 2) - 100),
                                    (int(frame_width / 2), int(frame_height / 2) + 100),
                                    (32, 183, 255), 4)
                elif median_diff_y < -5:
                    # 向下
                    cv2.arrowedLine(frame_canvas, (int(frame_width / 2), int(frame_height / 2) + 100),
                                    (int(frame_width / 2), int(frame_height / 2) - 100),
                                    (32, 183, 255), 4)
                if median_diff_x > 5:
                    # 向左
                    cv2.arrowedLine(frame_canvas, (int(frame_width / 2) - 100, int(frame_height / 2)),
                                    (int(frame_width / 2) + 100, int(frame_height / 2)),
                                    (32, 183, 255), 4)
                elif median_diff_x < -5:
                    # 向右
                    cv2.arrowedLine(frame_canvas, (int(frame_width / 2) + 100, int(frame_height / 2)),
                                    (int(frame_width / 2) - 100, int(frame_height / 2)),
                                    (32, 183, 255), 4)

            # 绘制按钮
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
