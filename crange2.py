#ライブラリ
import cv2 #映像処理
import numpy as np
import math

def find_circle_of_target_color(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = np.zeros(h.shape, dtype=np.uint8)
    if color == 0: #白
        mask[(s < 150) & (v > 150)] = 255
    elif color == 1: #黄
        mask[(h > 40) & (h < 70) & (s > 50) & (v > 50)] = 255
    elif color == 2: #緑
        mask[(h > 70) & (h < 140) & (s > 50) & (v > 50)] = 255
    elif color == 3: #青
        mask[(h > 150) & (h < 170) & (s > 50) & (v > 50)] = 255
    elif color == 4: #赤
        mask[(h > 250) & (h < 270) & (s > 50) & (v > 50)] = 255
    elif color == 5: #橙
        mask[(((h > 0) & (h < 40)) | ((h > 200) & (h < 360))) & (s > 50) & (v > 50)] = 255
    hsv_min = np.array([160,100,50])
    hsv_max = np.array([180,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

def color_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 色のHSVの値域
    hsv_min = np.array([160,100,50])
    hsv_max = np.array([180,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask


VIDEOFILE = 'sample1' #ビデオファイル名


def main():
    video = cv2.VideoCapture(VIDEOFILE+'.mp4')
    allframe = int(video.get(7)) #総フレーム数
    rate = int(video.get(5)) #フレームレート
    #print(allframe)
    for f in range(allframe):
        ret, frame = video.read()
        height, width, channels = frame.shape[:3]
        #print(width,height)
        
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
        cv2.rectangle(frame,(int(mean0[0]), int(mean0[1])),(int(mean1[0]), int(mean1[1])), color=(0, 0, 255), thickness=5)

        #cv2.fillPoly(frame, pts=find_circle_of_target_color(frame,0), color=(0,0,255))
        #contours = np.array( [ [50,50], [50,150], [150, 150], [150,50] ] )
        #cv2.fillPoly(frame, [find_circle_of_target_color(frame,0)], color=(0,255,0))
        mask = color_detect(frame)
        cv2.imshow("Mask", mask)
        mask = np.array(mask).reshape((-1,1,2)).astype(np.int32)
        cv2.bitwise_and((0,255,0), (255,0,0), mask=mask)
        #mask = np.array(mask).reshape((-1,1,2)).astype(np.int32)
        #cv2.FillConvexPoly(frame, mask, (0,255,0), lineType=8, shift=0)
        print(mask)
        
        '''
        circles0 = find_circle_of_target_color(frame,0) #白
        for circle in circles0:
            xy = list(circle[0:2] + circle[0:2] + circle[2:4])
            for i in range(2):
                xy[i] = xy[i] // 2
            point = list(circle[0:2])
            for i in range(2):
                point[i] -= xy[i]
            r = int(math.sqrt(point[0] ** 2 + point[1] ** 2) * 0.7)
            if xy[0] > mean0[0] and xy[1] > mean0[1] and xy[0] < mean1[0] and xy[1] < mean1[1] and r < width / 9 and r > width / 30:
                #cv2.circle(frame, (xy[0], xy[1]), r, color=(255, 0, 0), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(255, 255, 255), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 0), thickness=4)
        
        circles1 = find_circle_of_target_color(frame,1) #黄
        for circle in circles1:
            xy = list(circle[0:2] + circle[0:2] + circle[2:4])
            for i in range(2):
                xy[i] = xy[i] // 2
            point = list(circle[0:2])
            for i in range(2):
                point[i] -= xy[i]
            r = int(math.sqrt(point[0] ** 2 + point[1] ** 2) * 0.7)
            if xy[0] > mean0[0] and xy[1] > mean0[1] and xy[0] < mean1[0] and xy[1] < mean1[1] and r < width / 9 and r > width / 30:
                #cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 255, 0), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 255, 255), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 0), thickness=4)
        
        circles2 = find_circle_of_target_color(frame,2) #緑
        for circle in circles2:
            xy = list(circle[0:2] + circle[0:2] + circle[2:4])
            for i in range(2):
                xy[i] = xy[i] // 2
            point = list(circle[0:2])
            for i in range(2):
                point[i] -= xy[i]
            r = int(math.sqrt(point[0] ** 2 + point[1] ** 2) * 0.7)
            if xy[0] > mean0[0] and xy[1] > mean0[1] and xy[0] < mean1[0] and xy[1] < mean1[1] and r < width / 9 and r > width / 30:
                #cv2.circle(frame, (xy[0], xy[1]), r, color=(255, 255, 255), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 255, 0), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 0), thickness=4)

        circles3 = find_circle_of_target_color(frame,3) #青
        for circle in circles3:
            xy = list(circle[0:2] + circle[0:2] + circle[2:4])
            for i in range(2):
                xy[i] = xy[i] // 2
            point = list(circle[0:2])
            for i in range(2):
                point[i] -= xy[i]
            r = int(math.sqrt(point[0] ** 2 + point[1] ** 2) * 0.7)
            if xy[0] > mean0[0] and xy[1] > mean0[1] and xy[0] < mean1[0] and xy[1] < mean1[1] and r < width / 9 and r > width / 30:
                #cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 255, 255), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(255, 0, 0), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 0), thickness=4)
        
        circles4 = find_circle_of_target_color(frame,4) #赤
        for circle in circles4:
            xy = list(circle[0:2] + circle[0:2] + circle[2:4])
            for i in range(2):
                xy[i] = xy[i] // 2
            point = list(circle[0:2])
            for i in range(2):
                point[i] -= xy[i]
            r = int(math.sqrt(point[0] ** 2 + point[1] ** 2) * 0.7)
            if xy[0] > mean0[0] and xy[1] > mean0[1] and xy[0] < mean1[0] and xy[1] < mean1[1] and r < width / 9 and r > width / 30:
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 0), thickness=4)

        circles5 = find_circle_of_target_color(frame,5) #橙
        for circle in circles5:
            xy = list(circle[0:2] + circle[0:2] + circle[2:4])
            for i in range(2):
                xy[i] = xy[i] // 2
            point = list(circle[0:2])
            for i in range(2):
                point[i] -= xy[i]
            r = int(math.sqrt(point[0] ** 2 + point[1] ** 2) * 0.7)
            if xy[0] > mean0[0] and xy[1] > mean0[1] and xy[0] < mean1[0] and xy[1] < mean1[1] and r < width / 9 and r > width / 30:
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 150, 255), thickness=-1)
                cv2.circle(frame, (xy[0], xy[1]), r, color=(0, 0, 0), thickness=4)
        '''
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(rate)
        
        
    #video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


'''
show = bool(True) #画面表示の有無 True: 画面表示あり

#カラートラッキング
def color_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 色のHSVの値域
    hsv_min = np.array([150,100,0])
    hsv_max = np.array([190,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

print('OpenCV'+cv2.__version__)

VIDEOFILE = 'sample' #ビデオファイル名


#メイン
def main():

    #ビデオ
    video = cv2.VideoCapture(VIDEOFILE+'.mp4')

    data = []
    t = 0
    allframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) #総フレーム数

    #動画のカラートラッキング等の処理
    for i in range(allframe):
        # フレームを取得
        ret, frame = video.read()

        # 色検出
        mask = color_detect(frame)

        #データの追加
        data.append(mask)

        # 結果表示
        if show == True:
            #cv2.circle(frame, (x, y), 20, (0, 0, 255), 10)
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

        # qキーが押されたら途中終了
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    #np.savetxt("data.csv", np.array(data), delimiter=",")
    cv2.destroyAllWindows()

    print(data)


if __name__ == '__main__':
    main()

'''