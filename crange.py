#ライブラリ
import cv2 #映像処理
import numpy as np


import cv2
import numpy as np

def find_circle_of_target_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[(h < 190) & (h > 150) & (s > 128)] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        approx = cv2.convexHull(contour)
        circle = cv2.boundingRect(approx)
        circles.append(np.array(circle))
    return circles


VIDEOFILE = 'sample' #ビデオファイル名


def main():
    capture = cv2.VideoCapture(VIDEOFILE+'.mp4')
    while cv2.waitKey(30) < 0:
        _, frame = capture.read()
        circles = find_circle_of_target_color(frame)
        print(circles)
        for circle in circles:
            xy = list((circle[0:2])+list(circle[0:2] + circle[2:4]))
            for i in range(2):
                xy[i] = xy[i] // 2
            print(xy)
            cv2.circle(frame, xy[0], xy[1], 20, (0, 0, 255), thickness=2)
        cv2.imshow('red', frame)
    capture.release()
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