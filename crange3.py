#ライブラリ
import cv2 #映像処理
import numpy as np
import math
import copy

def color_detect(img,color):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 色のHSVの値域
    a = []
    b = []
    if color == 'white':
        a = [0,0,200]
        b = [180,50,255]
    elif color == 'yellow':
        a = [20,50,50]
        b = [50,255,255]
    elif color == 'green':
        a = [50,50,50]
        b = [70,255,255]
    elif color == 'blue':
        a = [70,50,50]
        b = [140,255,255]
    elif color == 'red':
        a = [160,50,50]
        b = [180,255,255]
    elif color == 'orange':
        a = [0,50,50]
        b = [20,255,255]
    hsv_min = np.array(a)
    hsv_max = np.array(b)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

def changecolor(height,width,dst,img_masked, color):
    a = ()
    if color == 'white':
        a = (255,255,255)
    elif color == 'yellow':
        a = (0,255,255)
    elif color == 'green':
        a = (0,255,0)
    elif color == 'blue':
        a = (255,0,0)
    elif color == 'red':
        a = (0,0,255)
    elif color == 'orange':
        a = (0,100,255)
    out = np.zeros((height, width, 3), dtype = "uint8") # 合成画像用の変数を作成
    for y in range(0, height):
        for x in range(0, width):
            if (img_masked[y][x] == 0).all(): #黒以外
                out[y][x] = dst[y][x] 
            else:
                out[y][x] = a
    return out

VIDEOFILE = 'sample3' #ビデオファイル名
fps = 30

def main():
    video = cv2.VideoCapture(VIDEOFILE+'.mp4')
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    allframe = int(video.get(7)) #総フレーム数
    rate = int(video.get(5)) #フレームレート
    #print(allframe)
    for f in range(allframe):
        ret, frame = video.read()

        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        dst = copy.copy(frame)
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            img_masked = cv2.bitwise_and(frame, frame, mask=mask)
            white = [255, 255, 255]
            black = [0, 0, 0]
            img_masked[np.where((img_masked != black).all(axis=2))] = white
            dst = changecolor(height,width,dst,img_masked,colorarray1[i])
        writer.write(dst)
        print(allframe,f)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
