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
    
def pathfunc():
    global VIDEOPATH
    VIDEOPATH = videopathbox.get()
    videopathbox.delete(0, tkinter.END)
    videopathbox.insert(tkinter.END,"OK")

def outpathfunc():
    global OUTPUTPATH
    OUTPUTPATH = outpathbox.get()
    outpathbox.delete(0, tkinter.END)
    outpathbox.insert(tkinter.END,"OK")

def setdefault():
    lightness.set(50)
    hue.set(0)
    resize.set(0.5)

def setmode():
    global mode
    if mode == False:
        mode = True
        modevar.set("Blue to White")
    else:
        mode = False
        modevar.set("White to Blue")

def inputvideo():
    global video, height, width, fps, fourcc, writer, allframe, percent, rate
    video = cv2.VideoCapture(VIDEOPATH)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    allframe = int(video.get(7))#総フレーム数
    rate = int(video.get(5)) #フレームレート
    video.release()
    
    explainlabel2 = tkinter.Label(root, text='Set Value and push One Frame Processing Button, If it is OK, Push Start Button')
    explainlabel2.pack()
    videolabel = tkinter.Label(root, text="height:"+str(height)+" width:"+str(width)+" framecnt:"+str(allframe)+" fps:"+str(fps))  #文字ラベル設定
    videolabel.pack() # 場所を指定　top, bottom, left, or right
    resizelabel = tkinter.Label(root, text='compression')  #文字ラベル設定
    resizelabel.pack() # 場所を指定　top, bottom, left, or right
    resizescale = tkinter.Scale(master=root, orient="horizontal", variable=resize, resolution=0.25, from_=0.5, to=1)
    resizescale.pack()
    lightnesslabel = tkinter.Label(root, text='lightness')  #文字ラベル設定
    lightnesslabel.pack() # 場所を指定　top, bottom, left, or right
    lightnessscale = tkinter.Scale(master=root, orient="horizontal", variable=lightness, from_=0, to=100)
    lightnessscale.pack()
    huelabel = tkinter.Label(root, text='hue')  #文字ラベル設定
    huelabel.pack() # 場所を指定　top, bottom, left, or right
    huescale = tkinter.Scale(master=root, orient="horizontal", variable=hue, from_=-20, to=20)
    huescale.pack()
    testframelabel = tkinter.Label(root, text='frame number')  #文字ラベル設定
    testframelabel.pack() # 場所を指定　top, bottom, left, or right
    testframescale = tkinter.Scale(master=root, orient="horizontal", variable=testframe, from_=1, to=allframe)
    testframescale.pack()
    setdefaultbutton = tkinter.Button(root, text='Set Default', command=setdefault)
    setdefaultbutton.pack()
    framebutton = tkinter.Button(root, text='One Frame Processing', command=testframefunc)
    framebutton.pack()
    modebutton = tkinter.Button(root, text='Mode', command=setmode)
    modebutton.pack()
    startbutton = tkinter.Button(root, text='Start', command=mainprocessing)
    startbutton.pack()
    stopbutton = tkinter.Button(root, text='Stop', command=stop)
    stopbutton.pack()
    videolabelvar = tkinter.StringVar()

def testframefunc():
    global testframe
    global video, height, width, fps, fourcc, writer, allframe, percent, rate
    video = cv2.VideoCapture(VIDEOPATH)
    frame_default = []
    for i in range(testframe.get()):
        ret, frame_default = video.read()
    dst = copy.copy(frame_default)
    frame = cv2.resize(frame_default, dsize=None, fx=resize.get(), fy=resize.get())
    colorarray0 = ['white','yellow','green','blue']
    colorarray1 = ['blue','green','white','yellow']
    if mode == True:
        tmp = colorarray0
        colorarray0 = colorarray1
        colorarray1 = tmp
    for i in range(len(colorarray0)):
        mask = color_detect(frame,colorarray0[i])
        if np.count_nonzero(mask) > 0:
            nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)            
        mask = cv2.resize(mask, dsize=None, fx=1 / resize.get(), fy=1 / resize.get())
        dst = changecolor(height,width,dst,mask,colorarray1[i])
    cv2.imshow('test frame',dst)
    #k = cv2.waitKey(rate)


def mainprocessing():
    global f, status
    if f == 0:
        global video, height, width, fps, fourcc, writer, allframe, percent, rate
        video = cv2.VideoCapture(VIDEOPATH)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        percent = 0
        print(VIDEOPATH)
        print(OUTPUTPATH)
        print(mode)
    global percentvar
    if f < allframe and status == True:
        ret, frame_default = video.read()
        dst = copy.copy(frame_default)
        frame = cv2.resize(frame_default, dsize=None, fx=resize.get(), fy=resize.get())
        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        if mode == True:
            tmp = colorarray0
            colorarray0 = colorarray1
            colorarray1 = tmp
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            if np.count_nonzero(mask) > 0:
                nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
                for j in range(len(data)):
                    if data[j][4] > 1 / 100 * width * height * resize.get() ** 2 or data[j][4] < 1 / 1000 * width * height * resize.get() ** 2:
                        for k in range(len(labelImages)):
                            if j in labelImages[k]:
                                for o in range(len(labelImages[k])):
                                    if labelImages[k][o] == j:
                                        mask[k][o] = 0
            mask = cv2.resize(mask, dsize=None, fx=1 / resize.get(), fy=1 / resize.get())
            dst = changecolor(height,width,dst,mask,colorarray1[i])
        #cv2.imshow('output',dst)
        #k = cv2.waitKey(rate)
        writer.write(dst)
        f += 1
        percent = int(f / allframe * 100)
        if f != allframe - 1:
            percentvar.set('done:' + str(percent)+'% (' + str(f) + '/' + str(allframe) + ')')
        root.after(1,mainprocessing)
    else:
        writer.release()
        status = False
        percentvar.set('Finished')
        #root.destroy()
            
            

def stop():
    global status
    status = False


root = tkinter.Tk()
root.title("Crange4 Setting")
root.geometry("500x800")
canvas = tkinter.Canvas(root, width = 100, height = 200)

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
testframe = tkinter.IntVar(master=root,value=1)
lightness = tkinter.IntVar(master=root,value=50)
hue = tkinter.IntVar(master=root,value=0)
resize = tkinter.DoubleVar(master=root,value=0.50)
VIDEOPATH = '' #ビデオパス
OUTPUTPATH = ''
mode = False #0: white to blue, 1: blue to white


explainlabel1 = tkinter.Label(root, text='Confirm Input & Output Path, then Input Video')
explainlabel1.pack()

pathlabel = tkinter.Label(root, text='Input Path')
pathlabel.pack()
videopathbox = tkinter.Entry(width=50)
videopathbox.pack()
pathbutton = tkinter.Button(root, text='Input Path Confirm', command=pathfunc)
pathbutton.pack()

virtuallabel1 = tkinter.Label(root, text='')
virtuallabel1.pack()

outpathlabel = tkinter.Label(root, text='Output Path')
outpathlabel.pack()
outpathbox = tkinter.Entry(width=50)
outpathbox.pack()
outpathbutton = tkinter.Button(root, text='Output Path Confirm', command=outpathfunc)
outpathbutton.pack()

virtuallabel2 = tkinter.Label(root, text='')
virtuallabel2.pack()

inputvideobutton = tkinter.Button(root, text='Input Video', command=inputvideo)
inputvideobutton.pack()

modevar = tkinter.StringVar()
modevar.set("White to Blue")
modelabel = tkinter.Label(root, textvariable=modevar)
modelabel.pack()

percentvar = tkinter.StringVar()
percentvar.set('processing percentage will be here')
label = tkinter.Label(root, textvariable=percentvar)
label.pack()

root.mainloop()