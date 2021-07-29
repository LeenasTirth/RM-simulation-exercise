import cv2
import numpy as np
import math
import random
import time

a = 0.785
omega = 1.884
b = 1.305
hook1 = []
hook2 = []
hook3 = []
huber_delta = 3.5


def getTheta(theta, t, dt, fai):
    global a
    global b
    global omega

    # 弧度
    v = a * math.sin(omega * t + fai) + b
    return theta + v * dt


def GetTheta(t):
    return (-0.785 / 1.884 * math.cos(1.884 * t + 4) + 1.305 * t) % (2 * math.pi)


def VSample(pretheta, posttheta, dt):
    dtheta = posttheta - pretheta
    if dtheta < 0:
        dtheta += 2 * math.pi
    return dtheta / dt


def process2(v_list, t_list, theta_list, fai):
    global a
    global b
    global omega
    global hook1
    global hook2
    global hook3
    global huber_delta

    # now = t_list[-1]
    # theta_now = theta_list[-1]
    theta_list = np.array(theta_list).copy()
    v_list = np.array(v_list).copy()
    t_list = np.array(t_list).copy()
    for i in range(5):
        vt = a * np.sin(omega * t_list + fai) + b
        dvt_dfai = a  * np.cos(omega * t_list + fai)
        dfai = (2 * (vt - v_list) * dvt_dfai).mean()
        fai = fai - 0.01 * dfai
    # theta_ = func(now, fai)
    theta_ = funcVec(t_list, fai)
    # theta0 = theta_now - theta_
    hook1 = theta_list.copy()
    hook2 = theta_.copy()
    hook3 = (theta_list - theta_)
    theta0 = ((theta_list - theta_) % (2 * math.pi)).mean()
    return fai, theta0



def func(t, fai):
    return (-0.785 / 1.884 * math.cos(1.884 * t + fai) + 1.305 * t) % (2 * math.pi)


def funcVec(t_list, fai):
    global a
    global b
    global omega
    return (-a / omega * np.cos(omega * t_list + fai) + b * t_list) % (2 * math.pi)


if __name__ == '__main__':
    img = np.zeros((1080 // 2, 1440 // 2, 3), np.uint8)
    img[:, :, :] = 255
    r = 150
    mid = (1440 // 4, 1080 // 4)
    cv2.circle(img, mid, r, (0, 0, 255), 0)
    theta = 0
    datalist = []
    fitlist = []
    t_list = []
    theta_list = []
    v_list = []
    fitpoint = mid
    fai = 0
    hat_theta = 0
    begin = 0
    end = 0
    start = time.time()
    t = start
    cnt = 0
    letgo = 10  # 从第十一个开始
    while True:
        now = time.time()
        dt = now - start - t  # 和前一刻的时间差
        t = now - start  # 真实时间

        theta = GetTheta(t)  # 模拟通过观测得到的theta

        dx = r * math.cos(theta)
        dy = r * math.sin(theta)
        point = (int(mid[0] + dx), int(mid[1] + dy))
        theta = (5 + theta) % (2 * math.pi) # 技巧
        datalist.append(point)  # 要追踪的目标点
        theta_list.append(theta)  # 保存观测到的角度
        t_list.append(t)
        if cnt == 0:
            cnt += 1
            cv2.waitKey(1000)
            continue
        v = VSample(theta_list[-2], theta_list[-1], dt)
        v_list.append(v)  # 保存观测到的角速度

        fai, theta0 = process2(v_list, t_list[1:], theta_list[1:], fai)
        theta0 = (theta0 - 5) % (math.pi * 2)
        hat_theta = func(t, fai) + theta0
        # hat_theta = getTheta(hat_theta, t, dt)
        dx = r * math.cos(hat_theta)
        dy = r * math.sin(hat_theta)
        fitpoint = (int(mid[0] + dx), int(mid[1] + dy))
        img_ = img.copy()
        cv2.circle(img_, point, 1, (255, 0, 0), 8)
        cv2.circle(img_, fitpoint, 1, (0, 255, 0), 10)

        cv2.imshow('circle', img_)
        k = cv2.waitKey(8)
        cnt += 1
        # t += 0.01
        # print(fai)
        print(fai)
        if k == 27:
            break

    # print(v_list)
    # print(hook1.tolist())
    # print("*" * 100)
    # print(hook2.tolist())
    # print(hook3.tolist())
