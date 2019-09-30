#ライブラリ
import cv2 #映像処理
import numpy as np
import math
import copy
import sys
import tkinter
from time import sleep

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
    

f = 0
video = []
height = 0
width = 0
fps = 0
fourcc = 0
writer = 0
allframe = 0
percent = 0
percentvar = 'a'
status = True

VIDEOPATH = '' #ビデオパス
resize = 0.5



root = tkinter.Tk()
root.title("Crange4 Setting")
root.geometry("500x500")
canvas = tkinter.Canvas(root, width = 100, height = 100)

videopathbox = tkinter.Entry(width=50)
videopathbox.insert(tkinter.END,"video path")
videopathbox.pack()

def pathfunc():
    global VIDEOPATH
    VIDEOPATH = videopathbox.get()
    videopathbox.delete(0, tkinter.END)
    videopathbox.insert(tkinter.END,"OK")
    print(VIDEOPATH)

pathbutton = tkinter.Button(root, text='OK', command=pathfunc)
pathbutton.pack()

compressionbox = tkinter.Entry(width=50)
compressionbox.insert(tkinter.END,"compression")
compressionbox.pack()

def compressionfunc():
    resize = compressionbox.get()
    compressionbox.delete(0, tkinter.END)
    compressionbox.insert(tkinter.END,"OK")
    print(resize)

compressionbutton = tkinter.Button(root, text='OK', command=compressionfunc)
compressionbutton.pack()

def mainprocessing():
    global f, status
    if f == 0:
        global video, height, width, fps, fourcc, writer, allframe, percent
        print(VIDEOPATH)
        video = cv2.VideoCapture(VIDEOPATH)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        allframe = int(video.get(7)) #総フレーム数
        percent = 0
    global percentvar

    if f < allframe and status == True:
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
        f += 1
        #print(f)
        if int(f / allframe * 100) != percent:
            percent = int(f / allframe * 100)
            #print(str(percent) + '%')
            percentvar.set('done:' + str(percent)+'%')
        root.after(1,mainprocessing)
    
    else:
        writer.release()
        status = False
        root.destroy()
            
            
    #cv2.destroyAllWindows()

startbutton = tkinter.Button(root, text='START', command=mainprocessing)
startbutton.pack()

def stop():
    global status
    status = False

startbutton = tkinter.Button(root, text='STOP', command=stop)
startbutton.pack()

percentvar = tkinter.StringVar()
percentvar.set('percentage')
label = tkinter.Label(root, textvariable=percentvar)
label.pack()


root.mainloop()

cv2.waitKey(1)