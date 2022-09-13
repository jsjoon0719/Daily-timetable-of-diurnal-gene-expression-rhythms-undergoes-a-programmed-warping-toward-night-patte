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


N_leaf = 4
 
resol = 15
cut = 3

folder = '1019'
fileList = os.listdir('temp/'+folder)
fileList.sort()

numberS = [1,2,3,4]
if 1:
	file = str(1) 

	dataStorm = load('temp/'+folder+'/leaf_'+str(1))
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
				yhat = nonNegative(yhat)
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

	def get_period(series, plotting = 0):
				tempLum = []
				tempTime = []
				peaks1, properties = find_peaks(series,distance = gap_needed)
				return timeList[peaks1[0]]

	def nonNegative(array):
		L = len(array)
		result = []
		prev = 0
		for a in array:
			if a > prev:
				result.append(a)
				prev = a
			else:
				result.append(prev)
		return result
def ploter(numberS, color):
	totalX = []
	totalY = []
	for num in numberS:
		dataStorm = load('temp/'+folder+'/leaf_'+str(num))
		(size_Y,size_X) = dataStorm[0][1].shape
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
			if time > 40:break
			if time > 5:
				xS.append(time)
				yS.append(lumSeries[t])
		plt.plot(yS)
		#plt.show()
		#plt.clf()
		result = (Process2PeriodFB(np.array(yS), xS, num))
		if result:
			print( result[0], result[1], get_lifespan(rgbSeries))
	return 0
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
#exit()

ploter([1,2,3,4,5,6,7,8,9,10,11,12],'r')
plt.show()
#exit()
import io
import cv2
for num in [1,2,3,4,5,6,7,8,9,10,11,12]:
	file = str(num) 

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
				yhat = nonNegative(yhat)
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

	def get_period(series, plotting = 0):
				tempLum = []
				tempTime = []
				for t in range(len(timeList)):
					time = timeList[t]
					if time > 65:
						break
					if time > 1:
						tempLum.append(series[t])
						tempTime.append(time)
				tempLum = regular(savgol_filter(tempLum, 51, 3))
				result = (Process2PeriodFB(np.array(tempLum), tempTime, plotting))
				if result: result=result[0]/(result[0]+result[1])
				if not result and 0:
					print ("godraw")
					(Process2PeriodFB(np.array(tempLum), tempTime, plotting))
				if not result:return 0
				return result

	def nonNegative(array):
		L = len(array)
		result = []
		prev = 0
		for a in array:
			if a > prev:
				result.append(a)
				prev = a
			else:
				result.append(prev)
		return result

	def makeTKimage(array):
		im_plt = plt.imshow(dataStorm[0][1])
		image1 = Image.fromarray(np.uint8( im_plt.get_cmap()(im_plt.get_array())*255))
		(size_Y,size_X) = np.shape(dataStorm[0][1])
		im = ImageTk.PhotoImage('RGB', image1.size)
		im.paste(image1)
		return im

	def clickNow(event):
		global scaleFactor, label_mainfig
		Y,X = (event.x/scaleFactor, event.y/scaleFactor)
		print (X,Y)
		x = int(X)
		y = int(Y)
		winS = 3
		lumX = []
		if dataStorm[0][1][y,x]:
				ymin = max(y-winS,0)
				ymax = min(size_Y,y+winS+1)
				xmin = max(x-winS,0)
				xmax = min(size_X,x+winS+1)
				lumSeries = []
				for t in range(len(dataStorm)):
					time = timeList[t]
					#if time < 60:
					if 1:
						lumSeries.append( np.nansum(dataStorm[t][1][ymin:ymax,xmin:xmax]))
						lumX.append(time)
				rgbSeries = [np.nansum(cybers[2][ymin:ymax,xmin:xmax][0])  for cybers in dataStorm]
		else:
			return 0
		plt.rcParams["figure.figsize"] = (6,2)

		get_period(lumSeries,1)
		plt.clf()
		#plt.figure(figsize=(6, 2))
		#plt.plot(lumX, lumSeries)
		#plt.xlim(0,60)
		#plt.show()
		#buf = io.BytesIO()
		#plt.savefig('temp.png')
		#buf.seek(0)
		test = cv2.imread('temp.png')
		#test = cv2.resize(test, dsize=(500,500), interpolation=cv2.INTER_AREA)
		#test = test.resize()
		test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
		test = Image.fromarray(test)
		test = ImageTk.PhotoImage(image=test)
		label_lumgraph.configure(image=test)
		label_lumgraph.image = test

		plt.cla()

		#plt.figure(figsize=(6, 2))
		half = get_lifespan(rgbSeries)
		yhat = regular( savgol_filter(rgbSeries, 51, 3))
		yhat = nonNegative(yhat)
		plt.plot(timeList, yhat)
		plt.axvline(half)
		
		plt.savefig('temp.png')
		plt.cla()
		test = cv2.imread('temp.png')
		#test = cv2.resize(test, dsize=(500,500), interpolation=cv2.INTER_AREA)
		#test = test.resize()
		test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
		test = Image.fromarray(test)
		test = ImageTk.PhotoImage(image=test)
		label_rgbgraph.configure(image=test)
		label_rgbgraph.image = test




		root.update()



	if 0:
	
		root = tk.Tk()
		root.geometry('1200x1000')
		#canvas = tk.Canvas(root)
		#canvas.pack()
		#im_plt = plt.imshow(dataStorm[0][1])
		image1 = Image.fromarray(dataStorm[0][1])
		(size_Y,size_X) = np.shape(dataStorm[0][1])
		scaleFactor = 500/max([size_Y,size_X])
		image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
		image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)

		image1 = image1.resize((int(scaleFactor*size_Y),int(scaleFactor*size_X)))
		print (size_Y,size_X,(int(scaleFactor*size_Y),int(scaleFactor*size_X)))
		im = ImageTk.PhotoImage(image1)
		#im.paste(image1)
		#test = canvas.create_image(0, 0, image=im)
		label_mainfig = tk.Label(root, image = im )
		label_mainfig.pack()
		label_mainfig.place(x=10,y=10)
		label_mainfig.bind('<Button 1>', clickNow)

		label_lumgraph = tk.Label(root)
		label_lumgraph.pack()
		label_lumgraph.place(x=520,y=10)


		label_rgbgraph = tk.Label(root)
		label_rgbgraph.pack()
		label_rgbgraph.place(x=520,y=500)

		tk.mainloop()	
	



	lifegram = np.zeros((size_Y,size_X))
	periodgram = np.zeros((size_Y,size_X))
	periodS = []
	tempList = []
	lifespanS = []
	minPeriod = 0
	maxPeriod = 1
	minLifespan = 0
	maxLifespan = 200
	for x in range(size_X):
		print(x,size_X)
		for y in range(size_Y):
			lifegram[y,x] = None
			periodgram[y,x] = None
			winS = 1
			if dataStorm[0][1][y,x]:
				ymin = max(y-winS,0)
				ymax = min(size_Y,y+winS+1)
				xmin = max(x-winS,0)
				xmax = min(size_X,x+winS+1)
				lumSeries = [ np.nansum(cybers[1][ymin:ymax,xmin:xmax]) for cybers in dataStorm]
				rgbSeries = [np.nansum(cybers[2][ymin:ymax,xmin:xmax][0])  for cybers in dataStorm]
				(lumSeries)
				lifespan = get_lifespan(rgbSeries)
				#if lifespan < minLifespan:continue
				if period < minPeriod and period:period = minPeriod
				if period > maxPeriod:
					print (period)
					period = maxPeriod
				if period:
					periodgram[y,x] = period
				if lifespan > minLifespan and lifespan < maxLifespan:
					lifegram[y,x] = lifespan
				if lifespan > maxLifespan:
					lifegram[y,x] = maxLifespan
				if lifespan < minLifespan:
					lifegram[y,x] = minLifespan
				if lifespan > minLifespan and lifespan < maxLifespan:
						periodS.append(period)
						lifespanS.append(lifespan)
						tempList.append([period,lifespan])
	periodgram[0,0] = minPeriod
	periodgram[0,1] = maxPeriod
	from scipy import stats
	print (stats.pearsonr(periodS,lifespanS))

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


	def outout(mat, ratio=0.01):
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
			interName = round(np.mean(tempx),2)
			returnDict[interName] = tempy
		return returnDict



	cmap = plt.cm.gist_rainbow
	cmap.set_bad(color='black')


	if 1:


		plt.figure(figsize=(10, 10))

		plt.subplot(1,2,1)
		cmap = plt.cm.gist_rainbow_r
		cmap.set_bad(color='black')
		#periodFig = outout(periodgram)
		periodFig = periodgram

		if 0:






			plt.imshow(periodFig, cmap=cmap, interpolation='nearest')
			plt.subplot(1,2,2)
			plt.colorbar()
			cmap = plt.cm.gist_rainbow
			cmap.set_bad(color='black')

			lifeFig = outout(lifegram)
			plt.imshow(lifeFig, cmap=cmap, interpolation='nearest')
			plt.show()
			plt.subplot(3,2,3)
			#plt.plot(lifespanS ,lifespan2period, 'x')
			l2p = divider(periodS,lifespanS,5)
			keyLifeS = list(l2p.keys())
			keyLifeS.sort()
			plt.clf()
			plt.boxplot([l2p[lifespan] for lifespan in keyLifeS], showfliers=False)
			plt.xticks([n+1 for n in range(len(keyLifeS))],keyLifeS)
			plt.tick_params(labelsize=20)
			plt.xlabel('Half-life(h)', fontsize=20)
			plt.ylabel('Period(h)', fontsize=20)
			plt.show()
			plt.subplot(3,2,4)
			p2l = divider(lifespanS,periodS,5)
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

			print ('Complete!')

			plt.savefig('CheckFigS/'+folder+'/'+file+'png')
			plt.cla()

		try:os.mkdir('temp2/'+folder)
		except:pass
		try:os.mkdir('CheckFigS/'+folder)
		except:pass

		dataPack = [dataStorm, lifegram, periodgram, timeList, 1, 1]
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