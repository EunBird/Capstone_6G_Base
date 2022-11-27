import matplotlib.pyplot as plt
import matplotlib.patches as pch
import numpy as np
import random
from cv2 import cv2
import os
import gc

color = 'w'
color1 = 'r'


def FigRect(ax, y, state):  # 画矩形
    long = state[0]
    wide = state[1]
    rect = pch.Rectangle(xy=(0, 0), width=long, height=wide, edgecolor='k', facecolor='b', fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.add_patch(rect)
    del ax,rect,y, state
    gc.collect()


def FigRect1(ax, y, state):
    Len = state[2]
    rect = pch.Rectangle(xy=(0, y - Len), width=0.05, height=Len, edgecolor=color1, facecolor=color1, fill=True,
                         linewidth=2)
    ax.add_patch(rect)
    del ax, rect,y, state
    gc.collect()


def FigRect2(ax, x, state):
    Len = state[2]
    rect = pch.Rectangle(xy=(x - Len, 0), width=Len, height=0.05, edgecolor=color1, facecolor=color1, fill=True,
                         linewidth=2)
    ax.add_patch(rect)
    del ax, rect,x, state
    gc.collect()


def FigRect3(ax, y, state):
    long = state[0]
    Len = state[2]
    rect = pch.Rectangle(xy=(long - 0.05, y - Len), width=0.05, height=Len, edgecolor=color1, facecolor=color1,
                         fill=True,
                         linewidth=2)
    ax.add_patch(rect)
    del ax, rect,y, state
    gc.collect()


def FigRect4(ax, x, state):
    wide = state[1]
    Len = state[2]
    rect = pch.Rectangle(xy=(x - Len, wide - 0.05), width=Len, height=0.05, edgecolor=color1, facecolor=color1,
                         fill=True,
                         linewidth=2)
    ax.add_patch(rect)
    del ax, rect,x, state
    gc.collect()


def FigCir(center, radius):  # 画圆
    circle1 = plt.Circle((center[0], center[1]), radius, color='k', fill=True, linewidth=2)
    plt.gcf().gca().add_artist(circle1)
    del circle1,center, radius
    gc.collect()


def FigLine(k, b, color, state):  # 画线
    long = state[0]
    x = np.arange(0, long, 0.01)
    y = k * x + b
    # plt.plot(x, y, color=color)


def node(k, b, center, SigLocation, state):  # 切线与障碍物的交点
    # SigLocation = [state[0] / 2, state[1] / 2]
    if k == 0:
        x = SigLocation[0]
        y = center[1]
    else:
        x = (center[0] / k + center[1] - b) / (k + 1 / k)
        y = k * x + b
    return x, y


def fill(Object, SigLocation, state):
    for i in range(len(Object)):
        if Object[i][0] - Object[i][2] > SigLocation[0]:
            Color1([Object[i][0], Object[i][1]], Object[i][2], SigLocation, state)
        elif Object[i][0] + Object[i][2] < SigLocation[0]:
            Color2([Object[i][0], Object[i][1]], Object[i][2], SigLocation, state)
        elif Object[i][0] + Object[i][2] >= SigLocation[0] and Object[i][0] - Object[i][2] <= SigLocation[0]:
            if Object[i][1] - Object[i][2] > SigLocation[1]:
                Color3([Object[i][0], Object[i][1]], Object[i][2], SigLocation, state)
            elif Object[i][1] - Object[i][2] < SigLocation[1]:
                Color4([Object[i][0], Object[i][1]], Object[i][2], SigLocation, state)
            else:
                print('fill error 1')
                print(state, Object)
        else:
            print('fill error')
            print(state, Object)


def Color1(center, radius, SigLocation, state):
    long = state[0]
    k1, k2, b1, b2 = Tangent(center, radius, 'k', [SigLocation[0], SigLocation[1]], state)
    x1, y1 = node(k1, b1, center, SigLocation, state)
    x2, y2 = node(k2, b2, center, SigLocation, state)
    k, b = Line(x1, y1, x2, y2)
    if x1 < x2:
        X1 = np.arange(x1, x2, 0.01)
        X2 = np.arange(x2, long, 0.01)
        Y1 = k * X1 + b
        Y2 = k1 * X1 + b1
        Y3 = k1 * X2 + b1
        Y4 = k2 * X2 + b2
        plt.fill_between(X1, Y1, Y2, color=color)
        plt.fill_between(X2, Y3, Y4, color=color)
        del k1, k2, b1, b2,x1,y1,x2,y2,X1,X2,Y1,Y2,Y3,Y4,k,b,center, radius, SigLocation, state
        gc.collect()
    elif x1 > x2:
        X1 = np.arange(x2, x1, 0.01)
        X2 = np.arange(x1, long, 0.01)
        Y1 = k * X1 + b
        Y2 = k2 * X1 + b2
        Y3 = k1 * X2 + b1
        Y4 = k2 * X2 + b2
        plt.fill_between(X1, Y1, Y2, color=color)
        plt.fill_between(X2, Y3, Y4, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, Y1, Y2, Y3, Y4, k, b,center, radius, SigLocation, state
        gc.collect()
    else:
        X1 = np.arange(x1, long, 0.01)
        Y1 = k1 * X1 + b1
        Y2 = k2 * X1 + b2
        plt.fill_between(X1, Y1, Y2, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, Y1, Y2, k, b,center, radius, SigLocation, state
        gc.collect()


def Color2(center, radius, SigLocation, state):
    long = state[0]
    wide = state[1]
    # SigLocation = [long / 2, wide / 2]
    k1, k2, b1, b2 = Tangent(center, radius, 'k', [SigLocation[0], SigLocation[1]], state)
    x1, y1 = node(k1, b1, center, SigLocation, state)
    x2, y2 = node(k2, b2, center, SigLocation, state)
    k, b = Line(x1, y1, x2, y2)
    if x1 < x2:
        X1 = np.arange(0, x1, 0.01)
        X2 = np.arange(x1, x2, 0.01)
        Y1 = k1 * X1 + b1
        Y2 = k2 * X1 + b2
        Y3 = k * X2 + b
        Y4 = k2 * X2 + b2
        plt.fill_between(X1, Y1, Y2, color=color)
        plt.fill_between(X2, Y3, Y4, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, Y1, Y2, Y3, Y4, k, b,center, radius, SigLocation, state
        gc.collect()
    elif x1 > x2:
        X1 = np.arange(0, x2, 0.01)
        X2 = np.arange(x2, x1, 0.01)
        Y1 = k1 * X1 + b1
        Y2 = k2 * X1 + b2
        Y3 = k * X2 + b
        Y4 = k1 * X2 + b1
        plt.fill_between(X1, Y1, Y2, color=color)
        plt.fill_between(X2, Y3, Y4, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, Y1, Y2, Y3, Y4, k, b,center, radius, SigLocation, state
        gc.collect()
    else:
        X1 = np.arange(0, x1, 0.01)
        Y1 = k1 * X1 + b1
        Y2 = k2 * X1 + b2
        plt.fill_between(X1, Y1, Y2, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, Y1, Y2, k, b,center, radius, SigLocation, state
        gc.collect()


def Color3(center, radius, SigLocation, state):
    long = state[0]
    wide = state[1]
    # SigLocation = [long / 2, wide / 2]
    k1, k2, b1, b2 = Tangent(center, radius, 'k', [SigLocation[0], SigLocation[1]], state)
    x1, y1 = node(k1, b1, center, SigLocation, state)
    x2, y2 = node(k2, b2, center, SigLocation, state)
    k, b = Line(x1, y1, x2, y2)
    if center[0] + radius > SigLocation[0] and center[0] - radius < SigLocation[0]:
        X1 = np.arange(0, x1, 0.01)
        X2 = np.arange(x1, x2, 0.01)
        X3 = np.arange(x2, long, 0.01)
        Y1 = k1 * X1 + b1
        Y2 = k * X2 + b
        Y3 = k2 * X3 + b2
        plt.fill_between(X1, Y1, wide, color=color)
        plt.fill_between(X2, Y2, wide, color=color)
        plt.fill_between(X3, Y3, wide, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, X3,Y1, Y2, Y3, k, b,center, radius, SigLocation, state
        gc.collect()
    elif center[0] + radius == SigLocation[0]:
        X1 = np.arange(0, x2, 0.01)
        X2 = np.arange(x2, x1, 0.01)
        Y1 = k2 * X1 + b2
        Y2 = k * X2 + b
        plt.fill_between(X1, Y1, wide, color=color)
        plt.fill_between(X2, Y2, wide, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2,Y1, Y2, k, b,center, radius, SigLocation, state
        gc.collect()
    elif center[0] - radius == SigLocation[0]:
        X1 = np.arange(SigLocation[0], x2, 0.01)
        X2 = np.arange(x2, long, 0.01)
        Y1 = k * X1 + b
        Y2 = k2 * X2 + b2
        plt.fill_between(X1, Y1, wide, color=color)
        plt.fill_between(X2, Y2, wide, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, Y1, Y2, k, b,center, radius, SigLocation, state
        gc.collect()
    else:
        print('Bottom error')
        print(state, center, radius)


def Color4(center, radius, SigLocation, state):
    long = state[0]
    wide = state[1]
    # SigLocation = [long / 2, wide / 2]
    k1, k2, b1, b2 = Tangent(center, radius, 'k', [SigLocation[0], SigLocation[1]], state)
    x1, y1 = node(k1, b1, center, SigLocation, state)
    x2, y2 = node(k2, b2, center, SigLocation, state)
    k, b = Line(x1, y1, x2, y2)
    if center[0] + radius > SigLocation[0] and center[0] - radius < SigLocation[0]:
        X1 = np.arange(0, x2, 0.01)
        X2 = np.arange(x2, x1, 0.01)
        X3 = np.arange(x1, long, 0.01)
        Y1 = k2 * X1 + b2
        Y2 = k * X2 + b
        Y3 = k1 * X3 + b1
        plt.fill_between(X1, 0, Y1, color=color)
        plt.fill_between(X2, 0, Y2, color=color)
        plt.fill_between(X3, 0, Y3, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, X3, Y1, Y2, Y3, k, b,center, radius, SigLocation, state
        gc.collect()
    elif center[0] + radius == SigLocation[0]:
        X1 = np.arange(0, x2, 0.01)
        X2 = np.arange(x2, x1, 0.01)
        Y1 = k2 * X1 + b2
        Y2 = k * X2 + b
        plt.fill_between(X1, 0, Y1, color=color)
        plt.fill_between(X2, 0, Y2, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, Y1, Y2, k, b,center, radius, SigLocation, state
        gc.collect()
    elif center[0] - radius == SigLocation[0]:
        X1 = np.arange(SigLocation[0], x2, 0.01)
        X2 = np.arange(x2, long, 0.01)
        Y1 = k * X1 + b
        Y2 = k2 * X2 + b2
        plt.fill_between(X1, 0, Y1, color=color)
        plt.fill_between(X2, 0, Y2, color=color)
        del k1, k2, b1, b2, x1, y1, x2, y2, X1, X2, Y1, Y2, k, b,center, radius, SigLocation, state
        gc.collect()
    else:
        print('Top error')
        print(state, center, radius)



def Tangent(center, radius, color, point, state):  # 圆的切线
    a = (point[0] - center[0]) ** 2 - radius ** 2
    b = radius ** 2 - (center[1] - point[1]) ** 2
    c = (point[0] - center[0]) * (center[1] - point[1])
    if a == 0:
        # plt.axvline(point[0], color='k')
        k1 = b1 = 0
        d = (center[1] - point[1]) / (center[0] - point[0])
        x = (center[1] - point[1]) / (1 / d + d) + point[0]
        y = d * (x - point[0]) + point[1]
        e = 2 * x - point[0]
        f = 2 * y - center[1]
        k2 = (f - point[1]) / (e - point[0])
        b2 = -point[0] * k2 + point[1]
        del a,b,c,d,x,y,e,f,center, radius, color, point, state
        gc.collect()
    else:
        k1 = ((b * a + c ** 2) ** (1 / 2) - c) / a
        k2 = (-(b * a + c ** 2) ** (1 / 2) - c) / a
        b1 = point[1] - point[0] * k1
        b2 = point[1] - point[0] * k2
        del center, radius, color, point, state
        gc.collect()
    #     FigLine(k1, b1, color, state)
    # FigLine(k2, b2, color, state)
    return k1, k2, b1, b2


def PLDistance(point, line):  # 点到直线距离
    dis = abs(point[1] - point[0] * line[0] - line[1]) / (1 + line[0] ** 2) ** (1 / 2)
    return dis


def Line(x1, y1, x2, y2):
    k1 = (y1 - y2) / (x1 - x2)
    b1 = -x2 * k1 + y2
    return k1, b1


def refLine(k, b, x, y):
    if k == 0:
        k1 = k
        b1 = b
    else:
        k1 = -k
        b1 = x * k + y
    return k1, b1


def LocObj(num, SigLocation, state, NumObj):
    long = state[0]
    wide = state[1]
    Object = []
    x=2.5
    y=2.5
    for i in range(num):
        radius = random.uniform(0.5,1.5)
        a = random.uniform(x, long - x)
        b = random.uniform(y, wide - y)
        while 1:
            sig=1
            if (SigLocation[0] - radius-0.2 < a < SigLocation[0] + radius+0.2 or SigLocation[1] - radius-0.2 < b < SigLocation[1] + radius+0.2 or a + radius == SigLocation[0] or a - radius == SigLocation[0] or b + radius == SigLocation[1] or b - radius == SigLocation[1]):
                sig = 0
            if sig == 0:
                radius = random.uniform(0.5,1.5)
                a = random.uniform(x, long - x)
                b = random.uniform(y, wide - y)
            else:
                break
        while 1:
            sig = 1
            for j in range(len(Object)):
                if ((Object[j][0] - a) ** 2 + (Object[j][1] - b) ** 2) ** 0.5 <= radius + Object[j][2] + 0.2 or (( SigLocation[0] - radius-0.2 < a < SigLocation[0] + radius+0.2 and SigLocation[1] - radius-0.2 < b <
                                                                                                                         SigLocation[1] + radius+0.2) or a + radius == SigLocation[0] or a - radius == SigLocation[0] or b + radius == SigLocation[1] or b - radius == SigLocation[1]):
                    sig = 0
            if sig == 0:
                radius = random.uniform(0.5,1.5)
                a = random.uniform(x, long - x)
                b = random.uniform(y, wide - y)
            else:
                break
        Object.append([a, b, radius])
    while num < NumObj:
        Object.append([0, 0, 0])
        num = num + 1
    return Object


def Count(name1):
    img1 = cv2.imread(name1)
    area = 0
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(image1, (5, 5), 0)
    ret31, th31 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = th31.shape#300X300
    for i in range(height):
        for j in range(width):
            if th31[i, j] == 0:#검정색
                area += 1
    # cv2.imshow("add", th31)
    # cv2.waitKey(0)
    del img1, image1,blur1,ret31, th31
    gc.collect()
    return area


def Count2(name1, name2, name3):
    img1 = cv2.imread(name1)
    img2 = cv2.imread(name2)
    img3 = cv2.imread(name3)
    area = 0
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(image1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(image2, (5, 5), 0)
    blur3 = cv2.GaussianBlur(image3, (5, 5), 0)
    ret31, th31 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret32, th32 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret33, th33 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = th31.shape
    for i in range(height):
        for j in range(width):
            if th31[i, j] == 0 or th32[i, j] == 0 or th33[i, j] == 0:
                area += 1
                th31[i, j] = 0
    cv2.imshow("add", th31)
    cv2.waitKey(0)
    del ret31, th31,ret32, th32,ret33, th33,image1,image2,image3,img1,img2,img3
    gc.collect()
    return area


def Count3(name1, name2, name3,name4, name5, name6,name7):
    img1 = cv2.imread(name1)
    img2 = cv2.imread(name2)
    img3 = cv2.imread(name3)
    img4 = cv2.imread(name4)
    img5 = cv2.imread(name5)
    img6 = cv2.imread(name6)
    img7 = cv2.imread(name7)
    area = 0
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    image4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    image5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    image6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
    image7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(image1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(image2, (5, 5), 0)
    blur3 = cv2.GaussianBlur(image3, (5, 5), 0)
    blur4 = cv2.GaussianBlur(image4, (5, 5), 0)
    blur5 = cv2.GaussianBlur(image5, (5, 5), 0)
    blur6 = cv2.GaussianBlur(image6, (5, 5), 0)
    blur7 = cv2.GaussianBlur(image7, (5, 5), 0)
    ret31, th31 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret32, th32 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret33, th33 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret34, th34 = cv2.threshold(blur4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret35, th35 = cv2.threshold(blur5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret36, th36 = cv2.threshold(blur6, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret37, th37 = cv2.threshold(blur7, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = th31.shape
    for i in range(height):
        for j in range(width):
            if th31[i, j] == 0 or th32[i, j] == 0 or th33[i, j] == 0 or th34[i, j] == 0 or th35[i, j] == 0 or th36[i, j] == 0 or th37[i, j] == 0:
                area += 1
                th31[i, j] = 0
    # cv2.imshow("add", th31)
    # cv2.waitKey(0)
    del ret31, th31,ret32, th32,ret33, th33,ret34, th34,ret35, th35,ret36, th36,ret37, th37,image1,image2,image3,image4,image5,image6,image7,img1,img2,img3,img4,img5,img6,img7
    gc.collect()
    return area

def Count4(name1, name2, name3,name4, name5):
    img1 = cv2.imread(name1)
    img2 = cv2.imread(name2)
    img3 = cv2.imread(name3)
    img4 = cv2.imread(name4)
    img5 = cv2.imread(name5)
    area = 0
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    image4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    image5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(image1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(image2, (5, 5), 0)
    blur3 = cv2.GaussianBlur(image3, (5, 5), 0)
    blur4 = cv2.GaussianBlur(image4, (5, 5), 0)
    blur5 = cv2.GaussianBlur(image5, (5, 5), 0)
    ret31, th31 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret32, th32 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret33, th33 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret34, th34 = cv2.threshold(blur4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret35, th35 = cv2.threshold(blur5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = th31.shape
    for i in range(height):
        for j in range(width):
            if th31[i, j] == 0 or th32[i, j] == 0 or th33[i, j] == 0 or th34[i, j] == 0 or th35[i, j] == 0:
                area += 1
                th31[i, j] = 0
    # cv2.imshow("add", th31)
    # cv2.waitKey(0)
    del ret31, th31,ret32, th32,ret33, th33,ret34, th34,ret35, th35,image1,image2,image3,image4,image5,img1,img2,img3,img4,img5
    gc.collect()
    return area


def drawfig(Object, SigLocation, state, name, sig):
    long = state[0]
    wide = state[1]
    Len = state[2]
    fig, ax = plt.subplots(figsize=(long, wide))
    FigRect(ax, 0, state)

    X = np.arange(0, long, 0.01)
    plt.fill_between(X, 0, wide, color='b')
    # FigCir(SigLocation, 0.1)
    if sig == 0:
        fill(Object, SigLocation, state)
    elif sig == 1:
        fill(Object, SigLocation, state)
        FigRect1(ax, state[3], state)
    elif sig == 2:
        fill(Object, SigLocation, state)
        FigRect2(ax, state[3], state)
    elif sig == 3:
        fill(Object, SigLocation, state)
        FigRect3(ax, state[3], state)
    elif sig == 4:
        fill(Object, SigLocation, state)
        FigRect4(ax, state[3], state)
    else:
        print('drawfig error')
    for i in range(len(Object)):
        FigCir([Object[i][0], Object[i][1]], Object[i][2])
    plt.xlim(xmin=0, xmax=long)
    plt.ylim(ymin=0, ymax=wide)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(name, dpi=30)
    #plt.show()
    plt.clf()
    plt.close()
    del fig,ax
    gc.collect()
