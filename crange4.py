#ライブラリ
import cv2 #映像処理
import numpy as np
import math
import copy
import sys
import tkinter

def color_detect(img,color):
    # HSV色空間に変換
    global lightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 色のHSVの値域
    a = []
    b = []
    if color == 'white':
        a = [max(0,0+hue.get()),0,int(200 * lightness.get() / 100)]
        b = [180+hue.get(),128,255]
    elif color == 'yellow':
        a = [20+hue.get(),100,int(200 * lightness.get() / 100)]
        b = [50+hue.get(),255,255]
    elif color == 'green':
        a = [50+hue.get(),100,int(200 * lightness.get() / 100)]
        b = [70+hue.get(),255,255]
    elif color == 'blue':
        a = [80+hue.get(),150,int(100 * lightness.get() / 100)]
        b = [140+hue.get(),255,255]
    elif color == 'red':
        a = [160+hue.get(),50,int(100 * lightness.get() / 100)]
        b = [min(180,180+hue.get()),255,255]
    elif color == 'orange':
        a = [0,100+hue.get(),int(200 * lightness.get() / 100)]
        b = [20+hue.get(),255,255]
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
    



root = tkinter.Tk()
root.title("Crange4 Setting")
root.geometry("500x500")
canvas = tkinter.Canvas(root, width = 100, height = 100)

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
testframe = 0
lightness = tkinter.IntVar(master=root,value=50)
hue = tkinter.IntVar(master=root,value=0)
resize = tkinter.IntVar(master=root,value=0.5)
VIDEOPATH = '' #ビデオパス
OUTPUTPATH = ''


videopathbox = tkinter.Entry(width=50)
videopathbox.insert(tkinter.END,"video path")
videopathbox.pack()

def pathfunc():
    global VIDEOPATH
    VIDEOPATH = videopathbox.get()
    videopathbox.delete(0, tkinter.END)
    videopathbox.insert(tkinter.END,"OK")

pathbutton = tkinter.Button(root, text='OK', command=pathfunc)
pathbutton.pack()


outpathbox = tkinter.Entry(width=50)
outpathbox.insert(tkinter.END,"output path")
outpathbox.pack()

def outpathfunc():
    global OUTPUTPATH
    OUTPUTPATH = outpathbox.get()
    outpathbox.delete(0, tkinter.END)
    outpathbox.insert(tkinter.END,"OK")
    
outpathbutton = tkinter.Button(root, text='OK', command=outpathfunc)
outpathbutton.pack()

def inputvideo():
    global video, height, width, fps, fourcc, writer, allframe, percent, rate
    video = cv2.VideoCapture(VIDEOPATH)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    allframe = int(video.get(7))#総フレーム数
    rate = int(video.get(5)) #フレームレート
    video.release()
    videolabelvar.set("height:"+str(height)+" width:"+str(width)+" framecnt:"+str(allframe)+" fps:"+str(fps))

inputvideobutton = tkinter.Button(root, text='input video', command=inputvideo)
inputvideobutton.pack()

videolabelvar = tkinter.StringVar()
videolabelvar.set("height:"+str(height)+" width:"+str(width)+" framecnt:"+str(allframe)+" fps:"+str(fps))
videolabel = tkinter.Label(root, textvariable=videolabelvar)  #文字ラベル設定
videolabel.pack() # 場所を指定　top, bottom, left, or right


resizelabel = tkinter.Label(root, text='compression')  #文字ラベル設定
resizelabel.pack() # 場所を指定　top, bottom, left, or right
resizescale = tkinter.Scale(master=root, orient="horizontal", variable=resize, resolution=0.1, from_=0.1, to=1)
resizescale.pack()

lightnesslabel = tkinter.Label(root, text='lightness')  #文字ラベル設定
lightnesslabel.pack() # 場所を指定　top, bottom, left, or right
lightnessscale = tkinter.Scale(master=root, orient="horizontal", variable=lightness, from_=0, to=100)
lightnessscale.pack()

huelabel = tkinter.Label(root, text='lightness')  #文字ラベル設定
huelabel.pack() # 場所を指定　top, bottom, left, or right
huescale = tkinter.Scale(master=root, orient="horizontal", variable=hue, from_=-20, to=20)
huescale.pack()

framebox = tkinter.Entry(width=50)
framebox.insert(tkinter.END,"test frame")
framebox.pack()

def testframefunc():
    global testframe
    testframe = int(framebox.get())
    #framebox.delete(0, tkinter.END)
    #framebox.insert(tkinter.END,"OK")
    global video, height, width, fps, fourcc, writer, allframe, percent, rate
    video = cv2.VideoCapture(VIDEOPATH)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    allframe = int(video.get(7))#総フレーム数
    frame_default = []
    for i in range(min(allframe, testframe)):
        ret, frame_default = video.read()

    dst = copy.copy(frame_default)
    frame = cv2.resize(frame_default, dsize=None, fx=resize.get(), fy=resize.get())

    colorarray0 = ['white','yellow','green','blue']
    colorarray1 = ['blue','green','white','yellow']
    for i in range(len(colorarray0)):
        mask = color_detect(frame,colorarray0[i])
        if np.count_nonzero(mask) > 0:
            nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)            
        mask = cv2.resize(mask, dsize=None, fx=1 / resize.get(), fy=1 / resize.get())
        dst = changecolor(height,width,dst,mask,colorarray1[i])
    cv2.imshow('test frame',dst)
    #k = cv2.waitKey(rate)

framebutton = tkinter.Button(root, text='OK', command=testframefunc)
framebutton.pack()


def mainprocessing():
    global f, status
    if f == 0:
        global video, height, width, fps, fourcc, writer, allframe, percent, rate
        video = cv2.VideoCapture(VIDEOPATH)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(OUTPUTPATH, fourcc, fps, (width, height))
        allframe = int(video.get(7))#総フレーム数
        rate = int(video.get(5)) #フレームレート
        percent = 0
        print(VIDEOPATH)
        print(OUTPUTPATH)
        print(resize)
    global percentvar

    if f < allframe and status == True:
        ret, frame_default = video.read()
        dst = copy.copy(frame_default)
        frame = cv2.resize(frame_default, dsize=None, fx=resize.get(), fy=resize.get())

        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            if np.count_nonzero(mask) > 0:
                nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
                
                for j in range(len(data)):
                    if data[j][4] > 1 / 100 * width * height * resize ** 2 or data[j][4] < 1 / 1000 * width * height * resize ** 2:
                        if i == 0:
                            for k in range(len(labelImages)):
                                if j in labelImages[k]:
                                    for o in range(len(labelImages[k])):
                                        if labelImages[k][o] == j:
                                            mask[k][o] = 0
                
                        #print(mask)
            mask = cv2.resize(mask, dsize=None, fx=1 / resize.get(), fy=1 / resize.get())
            dst = changecolor(height,width,dst,mask,colorarray1[i])
        writer.write(dst)
        #cv2.imshow('output',dst)
        #k = cv2.waitKey(rate)
        f += 1
        #print(f)
        percent = int(f / allframe * 100)
        #print(str(percent) + '%')
        percentvar.set('done:' + str(percent)+'% (' + str(f) + '/' + str(allframe) + ')')
        root.after(1,mainprocessing)
    
    else:
        #frame.release()
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