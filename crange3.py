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
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    allframe = int(video.get(7)) #総フレーム数
    rate = int(video.get(5)) #フレームレート
    for f in range(allframe):
        ret, frame = video.read()
        dst = copy.copy(frame)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,255,apertureSize = 3)

        lines = cv2.HoughLines(edges,1,np.pi/180,10)
        l = []
        for i in range(100):
            for rho,theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                l.append([x1,y1,x2,y2])
                #cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        xy = []
        xtimesy = []
        for i in range(len(l)):
            for j in range(i+1,len(l)):
                y2y1 = l[i][3] - l[i][1]
                Y2Y1 = l[j][3] - l[j][1]
                x2x1 = l[i][2] - l[i][0]
                X2X1 = l[j][2] - l[j][0]
                x1 = l[i][0]
                y1 = l[i][1]
                X1 = l[j][0]
                Y1 = l[j][1]
                if x2x1 * X2X1 != 0:
                    a = y2y1 / x2x1
                    b = Y2Y1 / X2X1
                    if a - b != 0 and (1 + a * b) / math.sqrt((1 + a ** 2) * (1 + b ** 2)) < math.cos(10 * math.pi / 180):
                        x = int((a * x1 - y1 - b * X1 + Y1) / (a - b))
                        y = int(a * (x - x1) + y1)
                        if x > 0 and y > 0 and x < width and y < height:
                            xy.append([x,y])
                            xtimesy.append([x*y,len(xtimesy)])
                            #cv2.circle(frame,(x,y),2,color=(255,0,0),thickness=-1)
        xtimesy.sort()
        a = []
        for i in range(len(xtimesy)):
            a.append(xtimesy[i][1])
        xsort = []
        ysort = []
        for i in range(len(xy)):
            xsort.append(xy[a[i]][0])
            ysort.append(xy[a[i]][1])
        mean0 = []
        mean1 = []
        additionx = int(np.std(xsort))
        additiony = int(np.std(ysort))
        tmp0 = 0
        tmp1 = 0
        weightsum = 0
        for i in range(2 * len(xsort) // 7):
            weight = 1
            if i > 0:
                weight = 1 / (((xsort[i-1]-xsort[i]) / 5) ** 2 + ((ysort[i-1]-ysort[i]) / 5) ** 2 + 0.1)
            tmp0 += int(xsort[i] * weight)
            tmp1 += int(ysort[i] * weight)
            weightsum += weight
            #print(weight)
            #cv2.circle(frame,(xsort[i],ysort[i]),int(weight),color=(255,0,0),thickness=2)
        mean0 = [max(tmp0 // weightsum - additionx, 0), max(tmp1 // weightsum - additiony,0)]

        tmp0 = 0
        tmp1 = 0
        weightsum = 0
        for i in range(3 * len(xsort) // 5, 5 * len(xsort) // 5):
            weight = 1 / (((xsort[i-1]-xsort[i]) / 5) ** 2 + ((ysort[i-1]-ysort[i]) / 5) ** 2 + 0.1)
            tmp0 += int(xsort[i] * weight)
            tmp1 += int(ysort[i] * weight)
            weightsum += weight
            #print(weight)
            #cv2.circle(frame,(xsort[i],ysort[i]),int(weight),color=(0,255,0),thickness=2)
        mean1 = [min(tmp0 // weightsum + additionx, width), min(tmp1 // weightsum + additiony, height)]

        #print(mean0,mean1)
        cv2.rectangle(dst,(int(mean0[0]), int(mean0[1])),(int(mean1[0]), int(mean1[1])), color=(0, 0, 255), thickness=5)


        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            dst = changecolor(height,width,dst,mask,colorarray1[i])
        #writer.write(dst)
            
        cv2.imshow('Frame',dst)
        k = cv2.waitKey(rate)
        
        print(allframe,f)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
