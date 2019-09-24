#ライブラリ
import cv2 #映像処理
import numpy as np
from scipy.interpolate import interp1d # scipyのモジュール
import math
import openpyxl #excelファイル読み込み
import matplotlib.pyplot as plt #グラフ作成
#from scipy.optimize import curve_fit 
from scipy import optimize #近似

show = bool(True) #画面表示の有無 True: 画面表示あり
outputpng = bool(True) #画像アウトプット True:あり
outputexcel = bool(False) #Excelへのアウトプット True:あり

#カラートラッキング
def color_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 色のHSVの値域
    hsv_min = np.array([160,100,50])
    hsv_max = np.array([180,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

print('OpenCV'+cv2.__version__)

VIDEOFILE = 'sample' #ビデオファイル名


#メイン
def main():

        if VIDEOFILE[h] != None:
            #ビデオ
            video = cv2.VideoCapture(VIDEOFILE[h]+'.mp4')
            
            data = []
            t = 0
            allframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) #総フレーム数

            #動画のカラートラッキング等の処理
            for i in range(allframe):
                # フレームを取得
                ret, frame = video.read()

                # 色検出
                mask = color_detect(frame)

                #最大面積の重心
                x, y = calc_max_point(mask)

                #単位変換
                x *= adjlen[h]
                y *= adjlen[h]

                #屈折処理
                x = refraction(x,h)

                #データの追加
                data.append([t, x, y])

                # 結果表示
                if show == True:
                    cv2.circle(frame, (int(x / adjlen[h]), int(y / adjlen[h])), 20, (0, 0, 255), 10)
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Mask", mask)

                t = video.get(cv2.CAP_PROP_POS_FRAMES) / FPS[h]

                # qキーが押されたら途中終了
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                
                #np.savetxt("data.csv", np.array(data), delimiter=",")
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()