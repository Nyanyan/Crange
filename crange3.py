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
    hsv_min = np.array([30,100,50])
    hsv_max = np.array([80,255,255])
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

        
        mask = color_detect(frame)
        #cv2.imshow("Mask", mask)
        #mask = np.array(mask).reshape((-1,1,2)).astype(np.int32)
        img_masked = cv2.bitwise_and(frame, frame, mask=mask)
        '''
        for x in range(height):
            for y in range(width):
                b, g, r = img_masked[x, y]
                if (b, g, r) == (255, 255, 255):
                    continue
                img_masked[x, y] = 0, 0, g
        '''
        color = [0, 0, 255]
        black = [0, 0, 0]
        img_masked[np.where((img_masked != black).all(axis=2))] = color
        
        dst = np.zeros((height, width, 3), dtype = "uint8") # 合成画像用の変数を作成
        for y in range(0, height):
            for x in range(0, width):
                if (img_masked[y][x] == 0).all(): # 「青・緑・赤」すべてが241以上なら
                    dst[y][x] = frame[y][x] # 隠れていない部分なので、「lena.jpg」の画素を代入
                else:
                    dst[y][x] = (0,0,255) # 隠れている部分なので、黒にする
        cv2.imshow('img',dst)

        #cv2.imshow("Show MASK Image", img_masked)
        
        #cv2.imshow("Show MASK Image", img_masked)
        #cv2.imwrite(frame, img_masked)
        '''
        img2gray = cv2.cvtColor(img_masked,cv2.COLOR_BGR2GRAY)
        img2gray.shape
        mask_inv = cv2.bitwise_not(img2gray)
        mask_inv.shape
        white_background = np.full(img_masked.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
        bk.shape
        
        fg = cv2.bitwise_or(img_masked, img_masked, mask=mask_inv)
        #final_roi = cv2.bitwise_or(frame,fg)
        
        clip_image(0, 0,fg,frame)
        '''
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
        #cv2.imshow("Frame", frame)
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