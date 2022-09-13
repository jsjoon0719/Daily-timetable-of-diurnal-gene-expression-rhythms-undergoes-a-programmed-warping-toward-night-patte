from matplotlib import pyplot as plt
import os
from tkinter import *
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import datetime
import cv2
from joblib import dump, load
from sklearn.svm import SVC
import sys
from scipy.signal import savgol_filter
import math


folder = 'RawData_0624'
fileList = os.listdir(folder) 
masterFolder = os.getcwd()
os.chdir(folder)
fileList.sort(key=os.path.getmtime)
os.chdir(masterFolder)
channelS = []
newfileList = []
fileRecordS = []
for file in fileList:
	if len(file.split('_')) > 2:  
		frontID = file.split('_')[0]+'&'+file.split('_')[1]
		if frontID in fileRecordS:continue
		fileRecordS.append(frontID)
	newfileList.append(file)
fileList = newfileList
for file in fileList:
	if 'test' in file:continue 
	channel_type = file.split('_')[1].split('.')[0]
	channelS.append(channel_type)
channelS = list(set(channelS))


with Image.open(folder+'/'+fileList[0]) as img:
	Time_start = datetime.datetime.strptime(img.tag[306][0], '%Y%m%d %H:%M:%S.%f')
	(Y,X) = np.shape(img)

T = (int(len(newfileList)/5))
C = len(channelS)
##450,660,730,white 우선

lumMatrix = np.zeros((Y,X,T))
rgbMatrix = np.zeros((Y,X,C-1,T))
hatSeries = np.zeros((T))
ztSeries = np.zeros((T))

for t in range(T):
	tempc = 0
	for c in range(C):
		file = (newfileList[5*t+c])
		#print (file)
		channel_type = file.split('_')[1].split('.')[0]
		if channel_type == 'LUM' or channel_type == 'lum':
			with Image.open(folder+'/'+file) as img:
				channel_type = file.split('_')[1].split('.')[0]
				#print (channel_type)
				time = datetime.datetime.strptime(img.tag[306][0], '%Y%m%d %H:%M:%S.%f')
				(Y,X) = np.shape(img)
				if channel_type == 'LUM' or channel_type == 'lum':
					lumMatrix[:,:,t] = img
					ztSeries[t] = ((time.day*24+time.hour+time.minute/60)-8)
					print (file, channel_type, t, time, ztSeries[t])
				else:
					rgbMatrix[:,:,tempc,t] = img
					tempc += 1

#dump([lum,ztSeries])


with open('fake.txt', 'w') as F:
	for x in range(16):
		for y in range(16):
			temp = lumMatrix[x*64:(x+1)*64,y*64:(y+1)*64,:].sum(axis=0).sum(axis=0)
			for t in temp:
				F.write(str(t)+'\t')
			F.write('\n')


print (Y,X)


