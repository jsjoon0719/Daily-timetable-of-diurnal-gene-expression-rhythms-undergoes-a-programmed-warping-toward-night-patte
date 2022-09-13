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
	import os


N_leaf = 4
resol = 15
cut = 3

folder = '1019/upperDisk'
fileList = os.listdir('temp/'+folder)
fileList.sort()

for file in fileList:

	dataStorm = load('temp/'+folder+'/'+file)
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

	def fsigmoid(x, a, b):
			return 1.0 / (1.0 + np.exp(a*(x-b)))

	def find_half(series, timeList = timeList):
				mid = (max(series)*0.5+min(series)*0.5)
				N = len(series)
				temp = []
				for n in range(N):
					temp.append([(series[n]-mid)**2,n])
				temp.sort()
				result = timeList[temp[0][1]]
				result = result/1
				return result

	def get_lifespan(series):
				yhat = savgol_filter(series, 51, 3)
				N = len(yhat)
				temp = []
				for n in range(N):
					temp.append([yhat[n],n])
				temp.sort(reverse=True)
				maxN = temp[0][1]
				newSeries = yhat[0:maxN]
				try:
					return find_half(newSeries)
				except:
					return 0

	def get_period(series):
				tempLum = savgol_filter(series, 51, 3)
				peaks, _ = find_peaks(tempLum, height=0, distance=50)
				peakTimeS = [timeList[peak] for peak in peaks]
				if len(peakTimeS) < 2:
					return 0
				period = peakTimeS[1]-peakTimeS[0]
				return period

	lifegram = np.zeros((size_Y,size_X))
	periodgram = np.zeros((size_Y,size_X))
	periodS = []
	tempList = []
	lifespanS = []
	minPeriod = 20
	maxPeriod = 30
	minLifespan = 60
	maxLifespan = 180
	for x in range(size_X):
		print(x,size_X)
		for y in range(size_Y):
			lifegram[y,x] = None
			periodgram[y,x] = None
			if dataStorm[0][1][y,x]:
				lumSeries = [cybers[1][y,x] for cybers in dataStorm]
				rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
				period = get_period(lumSeries)
				lifespan = get_lifespan(rgbSeries)
				if period > minPeriod and period < maxPeriod:
					periodgram[y,x] = period
				if lifespan > minLifespan and lifespan < maxLifespan:
					lifegram[y,x] = lifespan
				if period > minPeriod and period < maxPeriod:
					if lifespan > minLifespan and lifespan < maxLifespan:
						periodS.append(period)
						lifespanS.append(lifespan)
						tempList.append([period,lifespan])

	p2l = dict()
	l2p = dict()
	for [period,lifespan] in tempList:
		nnp = round(period,0)
		nl = round(lifespan,-1)
		if not nnp in p2l.keys():
			p2l[nnp] = []
		if not nl in l2p.keys():
			l2p[nl] = []
		print (nnp, nl)
		p2l[nnp].append(lifespan)
		l2p[nl].append(period)

	periodS = list(p2l.keys())
	print (periodS)
	periodS.sort()
	period2lifespan = []
	for period in periodS:
		period2lifespan.append(np.mean(p2l[period]))


	lifespanS = list(l2p.keys())
	lifespanS.sort()
	lifespan2period = []
	for lifespan in lifespanS:
		lifespan2period.append(np.mean(l2p[lifespan]))

	cmap = plt.cm.gist_rainbow
	cmap.set_bad(color='black')
	plt.subplot(2,2,1)
	plt.imshow(periodgram, cmap=cmap, interpolation='nearest')
	plt.subplot(2,2,2)
	plt.imshow(lifegram, cmap=cmap, interpolation='nearest')
	plt.subplot(2,2,3)
	#plt.plot(lifespanS ,lifespan2period, 'x')
	plt.boxplot([l2p[lifespan] for lifespan in lifespanS])
	plt.xticks([n+1 for n in range(len(lifespanS))],lifespanS)
	plt.subplot(2,2,4)
	plt.boxplot([p2l[period] for period in periodS])
	plt.xticks([n+1 for n in range(len(periodS))],periodS)
	#plt.plot(periodS ,period2lifespan, 'x')
	plt.show()