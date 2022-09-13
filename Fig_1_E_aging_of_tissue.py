if 1:
	import numpy as np
	import tkinter as tk
	from sklearn.svm import SVC
	from PIL import ImageTk, Image
	from matplotlib import pyplot as plt
	import sys
	from scipy.signal import savgol_filter
	from scipy.signal import find_peaks
	from joblib import dump, load
	import os
	from scipy.optimize  import curve_fit



	[dataStorm, lifegram, periodgram, timeList, p2l, l2p] = load('leaf_1')

	def fsigmoid(x, a, b):
		return 1 - 1.0 / (1.0 + np.exp(a*(x-b)))

	def monotonical(series):
			N = len(series)
			result = []
			result.append(series[0])
			for n in range(1,N):
				if series[n] > result[n-1]:
					result.append(series[n])
				else:
					result.append(result[n-1])
			return result

	def click(event):
		global scaleFactor, canvas
		x = int(event.x/scaleFactor)
		y = int(event.y/scaleFactor)
		rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
		lifespan = get_lifespan(rgbSeries)				
		yhat = regular(savgol_filter(monotonical(rgbSeries), 51, 3))
		popt, pcov = curve_fit(fsigmoid, range(len(timeList)), yhat, method='dogbox', p0 = [1,50])
		#curve_fit(fsigmoid, xdata, ydata, method='dogbox', bounds=([0., 600.],[0.01, 1200.]))
		yt = fsigmoid(range(len(timeList)), *popt)
		#print (y)


		plt.plot(timeList, regular(yt))
		plt.plot(timeList, regular(yhat))
		plt.axvline(lifespan)
		print(lifespan, x, y)
		plt.xlim(0,200)
		plt.show()


	def nonclick(x,y):
		global scaleFactor, canvas
		rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
		lifespan = get_lifespan(rgbSeries)				
		yhat = savgol_filter(rgbSeries, 51, 3)
		plt.plot(timeList, regular(yhat))
		plt.axvline(lifespan)
		print(lifespan)
		plt.show()




	def click2(event):
		global scaleFactor, canvas
		x = int(event.x/scaleFactor)
		y = int(event.y/scaleFactor)
		lumSeries = [(cybers[1][y,x]) for cybers in dataStorm]
		period, peaks, peakTimeS = get_period2(lumSeries)		
		yhat = savgol_filter(lumSeries, 51, 3)
		yhat = regular(yhat)
		plt.plot(timeList, regular(yhat))
		plt.xlim(15,60)
		#plt.plot(peaks)
		k = 0
		for pt in peaks:
			k += 1
			if k > 3:break
			plt.plot([timeList[pt]],[yhat[pt]],'o')
		print(period, x, y)
		plt.show()


	def rescaleMat(mat, factor = 100):
		flatter = mat.flatten()
		maxele = np.nanmax(flatter)
		minele = np.nanmin(flatter)
		#print (maxele,minele)
		#midele = np.median(flatter)
		#flatter.sort()
		#midele = flatter[int(len(flatter)*0.9)]
		return 1.0*(mat-minele)/(maxele-minele)*factor

	def regular(series):
		maxer = max(series)
		miner = min(series)
		result = [1.0*(ser-miner)/(maxer-miner) for ser in series]
		return result
	def find_half(series, timeList = timeList):
				mid = (max(series)*0.5+min(series)*0.5)
				N = len(series)
				temp = []
				for n in range(N):
					if series[n] >= max(series):break
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
				
				peaks, _ = find_peaks(tempLum, height=0, distance=30)
				peakTimeS = [timeList[peak] for peak in peaks]
				if len(peakTimeS) < 2:
					return 0
				period = peakTimeS[1]-peakTimeS[0]
				return period

	def get_period2(series):
				tempLum = savgol_filter(series, 51, 3)
				peaks, _ = find_peaks(tempLum, height=0, distance=30)
				peakTimeS = [timeList[peak] for peak in peaks]
				if len(peakTimeS) < 2:
					return 0
				period = peakTimeS[1]-peakTimeS[0]
				return period, peaks, peakTimeS



	size_Y, size_X = np.shape(dataStorm[0][1])

	cmap = plt.cm.gist_rainbow_r
	cmap.set_bad(color='black')


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

	def click(event):
		global scaleFactor, canvas
		x = int(event.x/scaleFactor)
		y = int(event.y/scaleFactor)
		rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
		lifespan = get_lifespan(rgbSeries)				
		yhat = regular(savgol_filter(monotonical(rgbSeries), 51, 3))
		popt, pcov = curve_fit(fsigmoid, range(len(timeList)), yhat, method='dogbox', p0 = [1,50])
		#curve_fit(fsigmoid, xdata, ydata, method='dogbox', bounds=([0., 600.],[0.01, 1200.]))
		yt = fsigmoid(range(len(timeList)), *popt)
		#print (y)


		plt.plot(timeList, regular(yt))
		plt.plot(timeList, regular(yhat))
		plt.axvline(lifespan)
		print(lifespan, x, y)
		plt.xlim(0,200)
		plt.show()

	def nonclick(x,y,c,label):
		global scaleFactor, canvas
		rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
		lim = 100
		for t in range(len(rgbSeries)):
			if t < lim:
				rgbSeries[t] = rgbSeries[lim]
		lifespan = get_lifespan(rgbSeries)				
		yhat = regular(savgol_filter(monotonical(rgbSeries), 51, 3))

		popt, pcov = curve_fit(fsigmoid, range(len(timeList)), yhat, method='dogbox', p0 = [1,50])
		#curve_fit(fsigmoid, xdata, ydata, method='dogbox', bounds=([0., 600.],[0.01, 1200.]))
		yt = fsigmoid(range(len(timeList)), *popt)
		#print (y)


		#plt.plot(timeList, regular(yt))
		plt.plot( timeList, regular(yhat), c, label = label)
		plt.axvline(lifespan, linestyle='--', color=c)
		print(lifespan, x, y)
		plt.tick_params(labelsize=20)
		plt.xlabel('Hours under continuous dark', fontsize=20)
		plt.ylabel('Reflectance of 430nm light', fontsize=20)
		plt.xlim(0,200)



	k = 0
	colorList = ['r', 'g', 'b']
	labelList= ['80h','100h','110h']
	for [x,y] in [[19, 18], [ 27, 32], [50, 57]]:
		nonclick(x,y,colorList[k],labelList[k])
		k+= 1
		if k == 3:continue
	plt.yticks([])
	plt.legend(loc='lower right', fontsize=17)
	plt.show()
	import matplotlib
	matplotlib.rcParams.update({'font.size': 50})
	if 1:

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

		lifeFig = outout(lifegram)
		colorMat = np.uint8((cmap(rescaleMat(lifeFig,1.0)))*255)
		for x in range(size_X):
			for y in range(size_Y):
				if not lifegram[y,x] > 30:
						colorMat[y,x] = [0,0,0,255]
				if not lifegram[y,x] < 110:
						colorMat[y,x] = [0,0,0,255]
		#plt.imshow(lifegram)
		#plt.show()
		img = Image.fromarray(colorMat)
		img = img.resize((sub_x,sub_y))
		img = ImageTk.PhotoImage(img)

		canvas.create_image(0,0,image=img, anchor = 'nw')
		root.bind('<Button 1>', click)
		root.mainloop()



	def build_gram(start,gap):
		global periodgram, lifegram
		lifegram = np.zeros((size_Y,size_X))
		periodgram = np.zeros((size_Y,size_X))
		periodS = []
		tempList = []
		lifespanS = []
		minPeriod = 18
		maxPeriod = 28
		minLifespan = 40
		maxLifespan = 200
		for x in range(size_X):
			print(x,size_X)
			for y in range(size_Y):
				lifegram[y,x] = None
				periodgram[y,x] = None
				if dataStorm[0][1][y,x]:
					#lumSeries = [np.nansum(cybers[1][y-1:y+2,x-1:x+2]) for cybers in dataStorm]
					#rgbSeries = [np.nansum(cybers[2][y-1:y+2,x-1:x+2][0])  for cybers in dataStorm]
					lumSeries = [(cybers[1][y,x]) for cybers in dataStorm[start:start+gap]]
					rgbSeries = [(cybers[2][y,x][0])  for cybers in dataStorm[start:start+gap]]
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

		minPeriod = 18
		maxPeriod = 28

		for pbg in range(10):
			print (pbg)
			build_gram(20+pbg*20,150)
			periodgram[0,0] = minPeriod
			periodgram[0,1] = maxPeriod
			periodFig = outout(periodgram)
			plt.imshow(periodFig, cmap=cmap)
			plt.savefig(str(pbg)+'.png')
			plt.clf()


	def nonclick2(x,y,c,minv,maxv,label):
		global scaleFactor, canvas
		lumSeries = [(cybers[1][y,x]) for cybers in dataStorm]
		period, peaks, peakTimeS = get_period2(lumSeries)		
		yhat = savgol_filter(lumSeries, 51, 3)

		newhat = []
		for t in range(len(timeList)):
			if t > 15  and t < 55:
				newhat.append(yhat[t])

		yhat = regular2(yhat,minv,maxv,newhat)
		plt.plot(timeList, yhat, c,label=label)
		plt.tick_params(labelsize=20)
		plt.xlim(15,55)
		#plt.plot(peaks)
		k = 0
		for pt in peaks:
			k += 1
			if k > 3:break
			plt.plot([timeList[pt]],[yhat[pt]],'o'+c)
			plt.axvline(timeList[pt], linestyle = '--', color = c)
		print(period, x, y)


	def regular2(series, minv, maxv,newhat):
		maxer = max(newhat)
		miner = min(newhat)
		result = [(maxv-minv)*(ser-miner)/(maxer-miner)+minv for ser in series]
		return result


	colorList = ['r', 'g', 'b']
	labelList= ['20h','24h','26h']

	k = 0
	for [x,y] in [[ 54, 37],[57, 35],[53,46]]:
		nonclick2(x,y,colorList[k],k*0.2,k*0.2+0.2,labelList[k])
		k+= 1
		if k == 3:continue
		plt.axhline(k*0.2+0.01,color='black')
	plt.ylim(0.03,0.63)
	plt.yticks([])
	import matplotlib
	matplotlib.rcParams.update({'font.size': 50})
	

	#matplotlib.rc('font', **font)
	#plt.rcParams['text.usetex'] = False
	#plt.ylim(0,1.4)
	plt.xlabel('Hours under constant dark', fontsize=20)
	plt.ylabel('Intensity of luminescence', fontsize=20)
	#plt.legend(loc='lower right', fontsize=17)
	plt.show()


	if 1:

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

		periodFig = outout(periodgram)

		colorMat = np.uint8((cmap(rescaleMat(periodFig,1.0)))*255)



		for x in range(size_X):
			for y in range(size_Y):
				if not periodgram[y,x] > 1:
					colorMat[y,x] = [0,0,0,255]
		#plt.imshow(lifegram)
		#plt.show()
		img = Image.fromarray(colorMat)
		img = img.resize((sub_x,sub_y))
		img = ImageTk.PhotoImage(img)

		canvas.create_image(0,0,image=img, anchor = 'nw')
		root.bind('<Button 1>', click2)
		root.mainloop()