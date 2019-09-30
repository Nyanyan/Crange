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
        a = [0,0,100]
        b = [180,50,255]
    elif color == 'yellow':
        a = [20,50,50]
        b = [50,255,255]
    elif color == 'green':
        a = [50,50,50]
        b = [70,255,255]
    elif color == 'blue':
        a = [80,50,50]
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
    dst[img_masked == 255] = a
    return dst

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

    for f in range(allframe):
        ret, frame = video.read()
        dst = copy.copy(frame)

        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            if np.count_nonzero(mask) > 0:
                nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
                print(data)
                for j in range(len(data)):
                    if data[j][4] > 1000000:
                        mask = mask + labelImages
                        #print(mask)
                        
                
            '''
            for j in range(width):
                cnt = 0
                start = 0
                for k in range(height):
                    if j > 0 and j < width - 2 and k > 0 and k < height - 2:
                        if mask[k][j] == 0:
                            tmp = 0
                            if mask[k-1][j] == 255:
                                tmp += 1
                            if mask[k+1][j] == 255:
                                tmp += 1
                            if mask[k][j-1] == 255:
                                tmp += 1
                            if mask[k][j+1] == 255:
                                tmp += 1
                            if tmp >= 3:
                                mask[k][j] = 255
                        if mask[k][j] == 255 and (mask[k-1][j] == 255 or mask[k][j-1] == 255):
                            if cnt == 0:
                                start = k
                            cnt += 1
                        else:
                            if cnt > min(width, height) / 5:
                                for o in range(start, k):
                                    mask[o][j] = 0
                            cnt = 0
                            start = 0
            '''
            dst = changecolor(height,width,dst,mask,colorarray1[i])
        writer.write(dst)
        cv2.imshow('Frames',dst)
        k = cv2.waitKey(rate)
        
        print(allframe,f)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
