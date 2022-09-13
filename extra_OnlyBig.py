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


N_leaf = 4
 
resol = 15
cut = 3

folder = '0815'
fileList = os.listdir('temp/'+folder)
fileList.sort()

numberS = [1,2,3,4]

def ploter(numberS, color):
	totalX = []
	totalY = []
	for num in numberS:

		dataStorm = load('temp/'+folder+'/leaf_'+str(num))
		(size_Y,size_X) = dataStorm[0][1].shape


		def regular(series):
			maxer = max(series)
			miner = min(series)
			result = [1.0*(ser-miner)/(maxer-miner) for ser in series]
			return result

		timeList = []
		lumSeries = []
		rgbSeries = []
		for [time, lum, rgb] in dataStorm:
				lumSeries.append(np.sum(lum))
				rgbSeries.append(np.sum(rgb))
				timeList.append(float(time))

		xS = []
		yS = []
		for t in range(len(timeList)):
			time = timeList[t]
			if time > 70:break
			if time > 5:
				xS.append(time)
				yS.append(lumSeries[t])
		yS = regular(yS)
		#plt.plot(xS,yS,color)
		if len(totalY) == 0:
			totalY = yS
		else:
			totalY = np.array(totalY)
			totalY += np.array(yS) 
	yhat = savgol_filter(totalY, 51, 3)
	plt.plot(xS,yhat,color)
	return regular(yhat)

old = ploter([1,2,3,4],'r')
ploter([5,6,7,8],'g')
young = ploter([33,34,35,36],'b')

plt.show()