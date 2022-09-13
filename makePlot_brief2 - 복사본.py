if 1:
	import numpy as np
	from joblib import dump, load
	import tkinter as tk
	from sklearn.svm import SVC
	from PIL import ImageTk, Image
	from matplotlib import pyplot as plt
	import sys
	from scipy.signal import savgol_filter
	from scipy.signal import find_peaks
	from joblib import dump, load
	import os
	from Useful_legacy import *
	from support_1 import *


N_leaf = 4
 
resol = 15
cut = 3

folder = '0625'
fileList = os.listdir('temp/Target/'+folder)
fileList.sort()

def ploter2(numberS, color, label, star, end):
	totalX = []
	totalY = []
	result2 = []
	for num in numberS:
		dataStorm = load('temp/Target/'+folder+'/brief_leaf_'+str(num))
		#(size_Y,size_X) = dataStorm[0][1].shape
		timeList = []
		lumSeries = []
		rgbSeries = []
		for [time, lum, rgb] in dataStorm:
					lumSeries.append(np.sum(lum))
					rgbSeries.append(np.sum(rgb))
					timeList.append(float(time)-0.55)
		lumSeries = regularizeCurve(lumSeries,timeList)
		plt.plot(lumSeries,'o-',color=color)
		xS = []
		yS = []
		start = 0
		pair = []
		for t in range(len(timeList)):
			time = timeList[t]
			if time > end:break
			if time > star:
				pair.append([time, lumSeries[t]])
		pair.sort()
		for [x,y] in pair:
			xS.append(x)
			yS.append(y)
		result2.append(yS)

		#plt.plot(xS, regular((yS)),color+'--')
		#plt.show()
		#result = (Process2PeriodFB(np.array(yS), xS, 0))
		result = 0
		if result:
			print( result[0], result[1], get_lifespan(rgbSeries))

	timegap = timeList[1]-timeList[0]
	yS = []
	errorS = []
	for t in range(len(xS)):
		yS.append(np.mean([y[t] for y in result2]))
		errorS.append(np.std([y[t] for y in result2])/2)
	curve = yS
	#curve = regular((smoothen(yS,5)))
	#curve = regular(yS)
	yS=curve
	#plt.plot(xS, yS,'o-',color=color,label=label)
	#plt.errorbar(xS,yS,yerr=errorS,color=color,capsize=3)
	plt.xlim(star,end)
	for k in range(10):
		plt.plot([0+k*24,8+k*24],[-0.1,-0.1],'black',linewidth=5)
		plt.plot([8+k*24,24+k*24],[-0.1,-0.1],'yellow',linewidth=5)
if 1:


	print ('go2')

	if 1:#0604Mutant
		titleS = ['WT_21DAS','ore1','toc1','elf4']
		startT = 0
		endT = 200
		showS = 10
		showE = 50

		if 1:
			ploter2(list(range(1,13)),'red',1,startT,endT)
			ploter2(list(range(13,25)),'green',1,startT,endT)
			ploter2([25],'gray',1,startT,endT)
			ploter2([26],'black',1,startT,endT)
			plt.xlim(showS,showE)

plt.show()