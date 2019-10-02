#ライブラリ
import cv2 #映像処理
import numpy as np
import math
import copy
import sys
import tkinter
import os
#from pydub import AudioSegment
#import ffmpeg
from moviepy.video.io.VideoFileClip import VideoFileClip
#import moviepy.editor as mp

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

def setdefault():
    lightness.set(50)
    hue.set(0)
    resize.set(0.5)
    deletenum.set(100)
    deletingfalg = True
    deletevar.set("Deleting Mode: True")
    mode = False
    modevar.set("Color Mode: White to Blue")

def setmode():
    global mode
    if mode == False:
        mode = True
        modevar.set("Color Mode: Blue to White")
    else:
        mode = False
        modevar.set("Color Mode: White to Blue")

def delete():
    global deleteflag
    if deleteflag == True:
        deleteflag = False
        deletevar.set("Deleting Mode: False")
        deletescale.config(state="disable")
    else:
        deleteflag = True
        deletevar.set("Deleting Mode: True")
        deletescale.config(state="active")

def inputvideo():
    global video, height, width, fps, fourcc, writer, allframe, percent, rate, VIDEOPATH, OUTPUTPATH, testframe
    VIDEOPATH = videopathbox.get()
    OUTPUTPATH = outpathbox.get()
    video = cv2.VideoCapture(VIDEOPATH)
    if video.isOpened() and OUTPUTPATH != '':
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        #allframe = int(video.get(CV_CAP_PROP_FRAME_COUNT))#総フレーム数
        #rate = int(video.get(5)) #フレームレート
        video.release()
        allframe = 0
        video2 = cv2.VideoCapture(VIDEOPATH)
        while True: #pyinstallerでなぜかフレーム数読み込みがうまくいかないので
            ret, frame = video2.read()
            if not ret:
                break  # 映像取得に失敗
            else:
                allframe += 1

        videolabelvar.set("height:"+str(height)+" width:"+str(width)+" framecnt:"+str(allframe)+" fps:"+str(fps))
        inputvideobutton.config(state="disable")
        outpathbox.config(state="disable")
        videopathbox.config(state="disable")
        explainlabel2.pack()
        videolabel.pack()
        resizelabel.pack()
        resizescale.pack()
        lightnesslabel.pack()
        lightnessscale.pack()
        huelabel.pack()
        huescale.pack()
        testframelabel.pack()
        testframe.set(1)
        testframescale = tkinter.Scale(master=root, orient="horizontal", variable=testframe, from_=1, to=allframe)
        testframescale.pack()
        deletelabel.pack()
        deletescale.pack()
        deletebutton.pack()
        modebutton.pack()
        setdefaultbutton.pack()
        framebutton.pack()
        startbutton.pack(fill='x')
        stopbutton.pack(fill='x')
    else:
        videoopenwarning = tkinter.Tk()
        videoopenwarning.title("Warning")
        videoopenwarning.geometry("100x100")
        warninglabel = tkinter.Label(videoopenwarning,text="Input Right Path")
        warninglabel.pack()


def testframefunc():
    global video, height, width, fps, fourcc, writer, allframe, percent, rate, testframe
    video = cv2.VideoCapture(VIDEOPATH)
    frame_default = []
    for i in range(testframe.get()):
        ret, frame_default = video.read()
    dst = copy.copy(frame_default)
    frame = cv2.resize(frame_default, dsize=None, fx=resize.get(), fy=resize.get())
    colorarray0 = ['white','yellow','green','blue']
    colorarray1 = ['blue','green','white','yellow']
    pre1 = 1 / deletenum.get() * width * height * resize.get() ** 2
    pre2 = 1 / resize.get()
    if mode == True:
        colorarray0, colorarray1 = colorarray1, colorarray0
    for i in range(len(colorarray0)):
        mask = color_detect(frame,colorarray0[i])
        if np.count_nonzero(mask) > 0:
            nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)   
            if deleteflag == True:
                for j in range(len(data)):
                        if data[j][4] > pre1 or data[j][4] < pre1 / 100:
                            for k in range(len(labelImages)):
                                if j in labelImages[k]:
                                    for o in range(len(labelImages[k])):
                                        if labelImages[k][o] == j:
                                            mask[k][o] = 0         
        mask = cv2.resize(mask, dsize=None, fx=1 / resize.get(), fy=1 / resize.get())
        dst = changecolor(height,width,dst,mask,colorarray1[i])
    cv2.imshow('test frame',dst)
    #k = cv2.waitKey(rate)


def mainprocessing():
    global f, status, percentvar
    if f == 0:
        global video, height, width, fps, fourcc, writer, allframe, percent, rate
        video = cv2.VideoCapture(VIDEOPATH)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))
        percent = 0
        resizescale.config(state="disable")
        lightnessscale.config(state="disable")
        huescale.config(state="disable")
        testframescale.config(state="disable")
        framebutton.config(state="disable")
        deletescale.config(state="disable")
        deletebutton.config(state="disable")
        modebutton.config(state="disable")
        setdefaultbutton.config(state="disable")
        startbutton.config(state="disable")
        print(VIDEOPATH)
        print(OUTPUTPATH)
        print(mode)
    pre1 = 1 / deletenum.get() * width * height * resize.get() ** 2
    pre2 = 1 / resize.get()
    if f < allframe and status == True:
        ret, frame_default = video.read()
        dst = copy.copy(frame_default)
        frame = cv2.resize(frame_default, dsize=None, fx=resize.get(), fy=resize.get())
        colorarray0 = ['white','yellow','green','blue']
        colorarray1 = ['blue','green','white','yellow']
        if mode == True:
            colorarray0, colorarray1 = colorarray1, colorarray0
        for i in range(len(colorarray0)):
            mask = color_detect(frame,colorarray0[i])
            if np.count_nonzero(mask) > 0:
                nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mask)
                if deleteflag == True:
                    for j in range(len(data)):
                        if data[j][4] > pre1 or data[j][4] < pre1 / 100:
                            for k in range(len(labelImages)):
                                if j in labelImages[k]:
                                    for o in range(len(labelImages[k])):
                                        if labelImages[k][o] == j:
                                            mask[k][o] = 0
            mask = cv2.resize(mask, dsize=None, fx=pre2, fy=pre2)
            dst = changecolor(height,width,dst,mask,colorarray1[i])
        #cv2.imshow('output',dst)
        #k = cv2.waitKey(rate)
        writer.write(dst)
        f += 1
        percent = int(f / allframe * 100)
        if f != allframe - 1:
            percentvar.set('Processing:' + str(percent)+'% (' + str(f) + '/' + str(allframe) + ')')
        root.after(1,mainprocessing)
    else:
        writer.release()
        status = False
        percentvar.set('Sound Encoding')
        stopsec = f / fps
        '''
        clip_input = AudioSegment.from_file(VIDEOPATH, "mp4")
        clip_input = sound[:stopsec * 1000]
        clip_output = ffmpeg.output("video.mp4", clip_input, OUTPUTPATH, vcodec="copy", acodec="aac")
        ffmpeg.run(clip_output)
        '''
        clip_input = VideoFileClip(VIDEOPATH).subclip(0,stopsec)
        clip_input.audio.write_audiofile('audio.mp3')
        clip_output = VideoFileClip('video.mp4').subclip(0,stopsec)
        clip_output.write_videofile(OUTPUTPATH, audio='audio.mp3')
        
        os.remove('video.mp4')
        os.remove('audio.mp3')
        percentvar.set('Finished')
        #root.destroy()

def stop():
    global status
    status = False


root = tkinter.Tk()
root.title("Crange4 Setting")
root.geometry("500x800")
canvas = tkinter.Canvas(root, width = 500, height = 800)
canvas.place(x=0,y=0)

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
testframe = tkinter.IntVar(master=root)
lightness = tkinter.IntVar(master=root,value=50)
hue = tkinter.IntVar(master=root,value=0)
resize = tkinter.DoubleVar(master=root,value=0.50)
deletenum = tkinter.DoubleVar(master=root,value=100)
VIDEOPATH = '' #ビデオパス
OUTPUTPATH = ''
mode = False #0: white to blue, 1: blue to white
deleteflag = False


explainlabel1 = tkinter.Label(root, text='Confirm Input & Output Path, then Input Video')
explainlabel1.pack()

pathlabel = tkinter.Label(root, text='Input Path')
pathlabel.pack()
videopathbox = tkinter.Entry(width=50)
videopathbox.pack()
#pathbutton = tkinter.Button(root, text='Input Path Confirm', command=pathfunc)
#pathbutton.pack()

virtuallabel1 = tkinter.Label(root, text='')
virtuallabel1.pack()

outpathlabel = tkinter.Label(root, text='Output Path')
outpathlabel.pack()
outpathbox = tkinter.Entry(width=50)
outpathbox.pack()
#outpathbutton = tkinter.Button(root, text='Output Path Confirm', command=outpathfunc)
#outpathbutton.pack()

virtuallabel2 = tkinter.Label(root, text='')
virtuallabel2.pack()

inputvideobutton = tkinter.Button(root, text='Input Video', command=inputvideo)
inputvideobutton.pack()

#canvas.create_rectangle((200,220),(300,300),outline='black')

statuslabel = tkinter.Label(root, text="===Status===")
statuslabel.pack()

videolabelvar = tkinter.StringVar()
videolabelvar.set("height:"+str(height)+" width:"+str(width)+" framecnt:"+str(allframe)+" fps:"+str(fps))
videolabel = tkinter.Label(root, textvariable=videolabelvar)  #文字ラベル設定
videolabel.pack()

modevar = tkinter.StringVar()
modevar.set("Color Mode: White to Blue")
modelabel = tkinter.Label(root, textvariable=modevar)
modelabel.pack()

deletevar = tkinter.StringVar()
deletevar.set("Deleting Mode: True")
deletestatelabel = tkinter.Label(root, textvariable=deletevar)
deletestatelabel.pack()

percentvar = tkinter.StringVar()
percentvar.set('processing percentage')
label = tkinter.Label(root, textvariable=percentvar)
label.pack()

statuslabel2 = tkinter.Label(root, text="============")
statuslabel2.pack()

explainlabel2 = tkinter.Label(root, text='Set Value and push One Frame Processing Button, If it is OK, Push Start Button')
resizelabel = tkinter.Label(root, text='compression')  #文字ラベル設定
resizescale = tkinter.Scale(master=root, orient="horizontal", variable=resize, resolution=0.25, from_=0.5, to=1)
lightnesslabel = tkinter.Label(root, text='lightness')  #文字ラベル設定
lightnessscale = tkinter.Scale(master=root, orient="horizontal", variable=lightness, from_=0, to=100)
huelabel = tkinter.Label(root, text='hue')  #文字ラベル設定
huescale = tkinter.Scale(master=root, orient="horizontal", variable=hue, from_=-20, to=20)
testframelabel = tkinter.Label(root, text='frame number')  #文字ラベル設定
testframescale = tkinter.Scale()
deletelabel = tkinter.Label(root, text='deleting size')  #文字ラベル設定
deletescale = tkinter.Scale(master=root, orient="horizontal", variable=deletenum, from_=1, to=500)
deletebutton = tkinter.Button(root, text='Delleting Mode', command=delete)
modebutton = tkinter.Button(root, text='Color Mode', command=setmode)
setdefaultbutton = tkinter.Button(root, text='Set Default', command=setdefault)
framebutton = tkinter.Button(root, text='One Frame Processing', command=testframefunc)
startbutton = tkinter.Button(root, text='Start', command=mainprocessing)
stopbutton = tkinter.Button(root, text='Stop', command=stop)


root.mainloop()