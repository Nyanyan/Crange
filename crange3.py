#ライブラリ
import cv2 #映像処理
import numpy as np
import math
import copy
import sys
import tkinter

def color_detect(img,color):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 色のHSVの値域
    a = []
    b = []
    if color == 'white':
        a = [0,0,100]
        b = [180,128,255]
    elif color == 'yellow':
        a = [20,100,100]
        b = [50,255,255]
    elif color == 'green':
        a = [50,100,100]
        b = [70,255,255]
    elif color == 'blue':
        a = [80,150,50]
        b = [140,255,255]
    elif color == 'red':
        a = [160,50,50]
        b = [180,255,255]
    elif color == 'orange':
        a = [0,100,100]
        b = [20,255,255]
    hsv_min = np.array(a)
    hsv_max = np.array(b)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

def changecolor(height,width,dst,img_masked, color):
    a = []
    if color == 'white':
        a = [255,255,255]
    elif color == 'yellow':
        a = [0,255,255]
    elif color == 'green':
        a = [0,255,0]
    elif color == 'blue':
        a = [255,0,0]
    elif color == 'red':
        a = [0,0,255]
    elif color == 'orange':
        a = [0,100,255]
    dst[img_masked > 0] = a
    return dst

VIDEOFILE = 'sample3.mp4' #ビデオファイル名
b = 200 #ステータスバーの上限

def main():
    video = cv2.VideoCapture('C:/home/Crange/' + VIDEOFILE)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    allframe = int(video.get(7)) #総フレーム数
    rate = int(video.get(5)) #フレームレート
    resize = 0.5
    cnt = 0

    for f in range(allframe):
        ret, frame_default = video.read()
        dst = copy.copy(frame_default)
        frame = cv2.resize(frame_default, dsize=None, fx=resize, fy=resize)

        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            if np.count_nonzero(mask) > 0:
                nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
                
                for j in range(len(data)):
                    #if i == 0:
                        #print(data[j][4])
                    if data[j][4] > 1 / 100 * width * height * resize ** 2 or data[j][4] < 1 / 1000 * width * height * resize ** 2:
                        if i == 0:
                            for k in range(len(labelImages)):
                                if j in labelImages[k]:
                                    for o in range(len(labelImages[k])):
                                        if labelImages[k][o] == j:
                                            mask[k][o] = 0
                
                        #print(mask)
            mask = cv2.resize(mask, dsize=None, fx=1 / resize, fy=1 / resize)
            dst = changecolor(height,width,dst,mask,colorarray1[i])
        writer.write(dst)
        #cv2.imshow('Frames',mask)
        #k = cv2.waitKey(rate)
        
        if f % (allframe // b) == 0:
            sys.stdout.write("\r")
            if cnt <= b:
                for i in range(cnt):
                    sys.stdout.write("=")
                for i in range(b - cnt):
                    sys.stdout.write(" ")
            sys.stdout.write("|")
            sys.stdout.flush()
            cnt+=1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
