import numpy as np
import matplotlib.pyplot as plt
from math import pi

a = 0.785
omega = 1.884
b = 1.305
huber = 1.5
fai = 3.45
lr = 0.1


def dataread(path):
    f = open(path)
    t = f.read()
    f.close()

    tt = t.split(',')
    tt = tt[:-1]
    t = [float(i) for i in tt]
    return t


def Huber(data):
    global a
    global omega
    global b
    global huber
    global fai
    global lr
    size = data.shape[0]
    v = data[:, 1]
    t = data[:, 0]
    for i in range(5000):
        vt = a * np.sin(omega * t + fai) + b
        dvt_dfai = a * np.cos(omega * t + fai)
        vt_v = vt - v
        branch1 = vt_v[abs(vt_v) <= huber].copy()
        br_dvt_dfai1 = dvt_dfai[abs(vt_v) <= huber].copy()
        br_dfai1 = branch1 * br_dvt_dfai1

        #branch2 = vt_v[abs(vt_v) > Huber].copy()
        br_dvt_dfai2_pos = dvt_dfai[vt_v > huber].copy()
        br_dfai2_pos = huber * br_dvt_dfai2_pos
        br_dvt_dfai2_neg = dvt_dfai[vt_v < -huber].copy()
        br_dfai2_neg = -huber * br_dvt_dfai2_neg
        dfai = np.hstack((br_dfai1, br_dfai2_pos, br_dfai2_neg)).mean()
        fai = fai - lr * dfai


def func(t):
    global a
    global omega
    global b
    global fai
    return a * np.sin(omega * t + fai) + b


def realfunc(t):
    global fai
    return 0.785 * np.sin(1.884 * t + fai) + 1.305


def Realtheta(t):
    global fai
    theta = -0.785 / 1.884 * np.cos(1.884 * t + fai) + 1.305 * t
    return theta % (2 * pi)


if __name__ == '__main__':
    v = dataread(
        "/home/lyl/vscode/ws/src/RoboMaster_parcket/CircleTracking/build/v.txt"
    )
    # print(len(t))
    # print(t[224])
    t = dataread(
        "/home/lyl/vscode/ws/src/RoboMaster_parcket/CircleTracking/build/t.txt"
    )
    theta = dataread(
        "/home/lyl/vscode/ws/src/RoboMaster_parcket/CircleTracking/build/theta.txt"
    )
    outangle = dataread(
        "/home/lyl/vscode/ws/src/RoboMaster_parcket/CircleTracking/build/outangle.txt"
    )
    outangle = np.array(outangle[1:]).reshape((-1, 1))
    outangle = (outangle * np.pi / 180.0) % (2.0 * np.pi)
    theta = np.array(theta[1:]).reshape((-1, 1))
    t = t[1:]
    v = np.array(v).reshape((-1, 1))
    t = np.array(t).reshape((-1, 1))
    data = np.hstack((t, v))
    print(len(t))
    print(len(v))
    Huber(data)
    print(fai)
    t_ = np.linspace(t.min() - 1, t.max() + 1, 5000)
    v_hat = func(t_)
    real_v = realfunc(t_)
    # x = np.arange(len(t))
    # t = np.array(t)

    plt.figure(figsize=(20, 15), dpi=80)
    plt.scatter(t, v)
    plt.plot(t_, v_hat)
    plt.plot(t_, real_v, 'r-')
    plt.show()
    realtheta = Realtheta(t_)
    plt.scatter(t, theta)
    plt.scatter(t, outangle, c='r')
    plt.plot(t_, realtheta)
    plt.show()

    ddd = t.copy()
    ddd = Realtheta(t)
    d = np.abs(ddd - outangle)
    t = t[d < 3]
    d = d[d < 3]
    plt.scatter(t, d)
    plt.show()

    # plt.show()
