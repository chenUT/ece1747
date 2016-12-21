import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw,ImageFont
from os import walk

import kmeansCL
import kmeansSeq

total_des_550 = None
total_des_350 = None

images = []
for (dirpath, dirnames, filenames) in walk("./imagemix"):
    images.extend(filenames)
    break

path = "./imagemix/"
for i in xrange(550):
    if ".DS_Store" not in images[i]: 
        imgPath = path + images[i]
        img=cv2.imread(imgPath)
        print("read: "+imgPath)
        res=cv2.resize(img,(250,250))
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray,None)
        if total_des_350 == None:
            total_des_550 = des
            total_des_350 = des
        else:
            if i < 350:
                total_des_350 = np.concatenate([total_des_350,des])
            if i < 550:
                total_des_550 = np.concatenate([total_des_550,des])

np.savetxt( 'mydata350.csv', total_des_350, fmt='%.2f', delimiter=',', newline='\n')           
np.savetxt( 'mydata550.csv', total_des_550, fmt='%.2f', delimiter=',', newline='\n')           


tdatas = [total_des_350, total_des_650]
tdatasLabel = ['total_des_350', 'total_des_550' ]

i = 0
for tdata in tdatas:
    print('start test: '+ tdatasLabel[i])
    cl_start = int(round(time.time() * 1000))
    c = kmeansCL.kmeans(tdata, 3)
    cl_end = int(round(time.time() * 1000))
    cl_time = cl_end - cl_start
    print('cl time: '+ str(cl_time))

    seq_start = int(round(time.time() * 1000))
    c2 = kmeansSeq.kmeans(tdata, 3)
    seq_end = int(round(time.time() * 1000))
    seq_time = seq_end - seq_start
    print('seq time: '+ str(seq_time))
    print('ratio '+ str(seq_time/cl_time))
    i +=1;
