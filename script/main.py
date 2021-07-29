import cv2
import numpy as np
import math
import random


def GetTheta(t):
    theta = -0.785 / 1.884 * math.cos(1.884 * t +
                                      4.0) + 1.305 * t + random.gauss(0, 0.01)
    return theta % (2.0 * math.pi)  # (tmp - float(int(tmp))) * 2 * math.pi


def process(thetalist, t_list, fai, theta0):
    thetalist = np.array(thetalist).copy()
    t_list = np.array(t_list).copy()
    for i in range(10):
        dtheta_dfai = 0.785 / 1.884 * np.sin(1.884 * t_list + fai)
        dfai = (2 * ((-0.785 / 1.884 * np.cos(1.884 * t_list + fai) +
                      1.305 * t_list + theta0) - thetalist) *
                dtheta_dfai).mean()
        dtheta0 = (2 * ((-0.785 / 1.884 * np.cos(1.884 * t_list + fai) +
                         1.305 * t_list + theta0) - thetalist)).mean()
        fai = fai - 0.01 * dfai
        theta0 = theta0 - 0.01 * dtheta0
    return fai, theta0


def process2(thetalist, t_list, datalist, fai, r, center):
    thetalist = np.array(thetalist).copy()
    t_list = np.array(t_list).copy()
    datalist = np.array(datalist).copy()
    for i in range(10):
        dtheta_dfai = 0.785 / 1.884 * np.sin(1.884 * t_list + fai)
        theta_t = -0.785 / 1.884 * np.cos(1.884 * t_list +
                                          fai) + 1.305 * t_list 
        dx_dfai = -r * np.sin(theta_t) * dtheta_dfai
        dy_dfai = r * np.cos(theta_t) * dtheta_dfai
        x_t = r * np.cos(theta_t) + center[0]
        y_t = r * np.sin(theta_t) + center[1]
        dfai = (2 * (x_t - datalist[:, 0].reshape(-1, 1)) * dx_dfai + 2 *
                (y_t - datalist[:, 1].reshape(-1, 1)) * dy_dfai).mean()

        fai = fai - 0.01 * dfai
    return fai


def func(t, fai, theta0):
    return -0.785 / 1.884 * math.cos(1.884 * t + fai) + 1.305 * t


if __name__ == '__main__':
    img = np.zeros((1080 // 2, 1440 // 2, 3), np.uint8)
    img[:, :, :] = 255
    r = 150
    mid = (1440 // 4, 1080 // 4)
    cv2.circle(img, mid, r, (0, 0, 255), 0)
    t = 0
    theta = 0
    datalist = []
    fitlist = []
    t_list = []
    theta_list = []
    fitpoint = mid
    fai = 0
    theta0 = 0
    max_len = 6000
    end_cond = 1e-5
    last_fai = 1e15
    last_theta0 = 1e15
    while True:
        theta = GetTheta(t)
        # theta = getAngle(theta)
        dx = r * math.cos(theta)
        dy = r * math.sin(theta)
        point = (int(mid[0] + dx), int(mid[1] + dy))
        datalist.append(point)
        fitlist.append(fitpoint)

        if len(t_list) >= max_len:
            t_list = t_list[1:]
            theta_list = theta_list[1:]
            print("ok!")
        t_list.append(t)
        theta_list.append(theta)
        # fai = process(datalist, t_list, theta_list, r, mid, fai)
        if abs(last_fai - fai) > end_cond or abs(last_theta0 -
                                                 theta0) > end_cond:
            #last_fai, last_theta0 = fai, theta0
            #fai, theta0 = process(theta_list, t_list, fai, theta0)
            fai = process2(theta_list, t_list, datalist, fai, r, mid)
            #print(abs(last_theta0 - theta0), abs(last_fai - fai))
        else:
            print("well")

        # fitpoint = func(t, fai)
        Ptheta = func(t, fai, theta0)
        fitpoint = (int(r * math.cos(Ptheta) + mid[0]),
                    int(r * math.sin(Ptheta) + mid[1]))

        img_ = img.copy()
        cv2.circle(img_, point, 1, (255, 0, 0), 8)
        cv2.circle(img_, fitpoint, 1, (0, 255, 0), 10)

        cv2.imshow('circle', img_)
        k = cv2.waitKey(1)
        t += 0.01
        print(fai)
        # print(theta0)
        if k == 27:
            break

    # print(fai)
    # print(theta_list)
    # print(len(theta_list))
