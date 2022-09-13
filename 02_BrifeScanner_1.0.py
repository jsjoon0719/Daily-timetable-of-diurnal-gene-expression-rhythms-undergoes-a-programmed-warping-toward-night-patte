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

folder = 'RawData_0624'
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
	maxPeriod = 28
	minLifespan = 40
	maxLifespan = 200
	for x in range(size_X):
		print(x,size_X)
		for y in range(size_Y):
			lifegram[y,x] = None
			periodgram[y,x] = None
			if dataStorm[0][1][y,x]:
				lumSeries = [ np.nansum(cybers[1][y-1:y+2,x-1:x+2]) for cybers in dataStorm]
				rgbSeries = [np.nansum(cybers[2][y-1:y+2,x-1:x+2][0])  for cybers in dataStorm]
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

	if 0:
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


	def outout(mat, ratio=0.05):
		flatter = mat.flatten()
		temp = []
		for value in flatter:
			if value > 1:
				temp.append(value)
		temp.sort()
		tempN = len(temp)
		lower = temp[int(tempN*ratio)]
		upper = temp[int(tempN*(1-ratio))]
		(tempY, tempX) = np.shape(mat)
		resultMat = np.zeros((tempY,tempX))
		for y in range(tempY):
			for x in range(tempX):
				resultMat[y,x] = None
				if mat[y,x] > 1:
					value = mat[y,x]
					if value > upper:
						value = upper
					if value < lower:
						value = lower
					resultMat[y,x] = value
		return resultMat


	def divider(Xlist, Ylist, iN):
		xn = len(Xlist)
		temp = Xlist[:]
		temp.sort()
		threshould = [temp[int(xn/iN*i)] for i in range(1,iN)]
		newX = []
		newY = []
		resultDict = dict()
		for n in range(xn):
			x = Xlist[n]
			y = Ylist[n]
			interver = 0
			for i in range(iN-1):
				if threshould[i] < x:
					interver = i+1
			if not interver in resultDict.keys():
				resultDict[interver] = [[],[]]
			resultDict[interver][0].append(x)
			resultDict[interver][1].append(y)
		returnDict = dict()
		for i in range(iN):
			[tempx, tempy] = resultDict[i]
			interName = round(np.mean(tempx),1)
			returnDict[interName] = tempy
		return returnDict



	cmap = plt.cm.gist_rainbow
	cmap.set_bad(color='black')


	if 1:
		plt.figure(figsize=(10, 10))

		plt.subplot(3,2,1)
		cmap = plt.cm.gist_rainbow_r
		cmap.set_bad(color='black')
		periodFig = outout(periodgram)
		plt.imshow(periodFig, cmap=cmap, interpolation='nearest')
		plt.subplot(3,2,2)
		cmap = plt.cm.gist_rainbow
		cmap.set_bad(color='black')

		lifeFig = outout(lifegram)
		plt.imshow(lifeFig, cmap=cmap, interpolation='nearest')
		plt.subplot(3,2,3)
		#plt.plot(lifespanS ,lifespan2period, 'x')
		l2p = divider(lifespanS,periodS,5)
		keyLifeS = list(l2p.keys())
		keyLifeS.sort()
		plt.boxplot([l2p[lifespan] for lifespan in keyLifeS], showfliers=False)
		plt.xticks([n+1 for n in range(len(keyLifeS))],keyLifeS)
		plt.subplot(3,2,4)
		p2l = divider(periodS,lifespanS,5)
		keyPeriodS = list(p2l.keys())
		keyPeriodS.sort()

		plt.boxplot([p2l[period] for period in keyPeriodS], showfliers=False)
		plt.xticks([n+1 for n in range(len(keyPeriodS))],keyPeriodS)
		#plt.plot(periodS ,period2lifespan, 'x')
		plt.subplot(3,2,5)
		lumSeries = []
		rgbSeries = []
		for [time, lum, rgb] in dataStorm:
			lumSeries.append(np.nansum(lum))
			rgbSeries.append(np.nansum(rgb))
		lumSeries = savgol_filter(lumSeries, 51, 3)
		rgbSeries = savgol_filter(rgbSeries, 51, 3)
		plt.plot(timeList, regular(lumSeries),'r',label='lum')
		plt.plot(timeList, regular(rgbSeries),'b',label='rgb')
		plt.legend(loc='upper right', fontsize=17)
		period = get_period(lumSeries)
		lifespan = get_lifespan(rgbSeries)
		plt.subplot(3,2,6)
		plt.text(0,0.1,'period: ' + str(round(period,1))+'h')
		plt.text(0,0.2,'lifespan: ' + str(round(lifespan,1))+'h')
		plt.text(0,0.3,'leaf#: ' + file)
		plt.xlim(-0.1,0.2)
		plt.axis('off')

		try:os.mkdir('temp2/'+folder)
		except:pass
		try:os.mkdir('CheckFigS/'+folder)
		except:pass

		print ('Complete!')

		plt.savefig('CheckFigS/'+folder+'/'+file+'png')
		plt.cla()

		dataPack = [dataStorm, lifegram, periodgram, timeList, p2l, l2p]
		dump(dataPack, 'temp2/'+folder+'/'+file)
		print (file, 'DONE!')





	def rescaleMat(mat, factor = 100):
		flatter = mat.flatten()
		maxele = np.nanmax(flatter)
		minele = np.nanmin(flatter)
		#print (maxele,minele)
		#midele = np.median(flatter)
		#flatter.sort()
		#midele = flatter[int(len(flatter)*0.9)]
		return 1.0*(mat-minele)/(maxele-minele)*factor



	def click(event):
		global scaleFactor, canvas
		x = int(event.x/scaleFactor)
		y = int(event.y/scaleFactor)
		rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
		lifespan = get_lifespan(rgbSeries)
		plt.plot(timeList, regular(rgbSeries))
		plt.axvline(lifespan)
		print(lifespan)
		plt.show()


	if 0:
		from tkinter import *
		from PIL import Image, ImageTk, ImageEnhance
		from matplotlib import cm
		maxSize = 800
		root = Tk()
		root.title("Cropper")
		root.geometry(str(maxSize)+'x'+str(maxSize))
		scaleFactor = maxSize/max([size_Y,size_X])
		(sub_x, sub_y) = (int(size_X*scaleFactor),int(size_Y*scaleFactor))
		canvas = Canvas(root, width = sub_x, height = sub_y)
		canvas.pack()

		colorMat = np.uint8((cmap(rescaleMat(lifegram,1.0)))*255)
		for x in range(size_X):
			for y in range(size_Y):
				if not lifegram[y,x] > 30:
					colorMat[y,x] = [0,0,0,255]
		#plt.imshow(lifegram)
		#plt.show()
		img = Image.fromarray(colorMat)
		img = img.resize((sub_x,sub_y))
		img = ImageTk.PhotoImage(img)

		canvas.create_image(0,0,image=img, anchor = 'nw')
		root.bind('<Button 1>', click)
		root.mainloop()