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



	[dataStorm, lifegram, periodgram, timeList, p2l, l2p] = load('leaf_21')


	def click(event):
		global scaleFactor, canvas
		x = int(event.x/scaleFactor)
		y = int(event.y/scaleFactor)
		rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
		rgbSeries = monotonical(rgbSeries)
		rgbSeries = regular(rgbSeries)
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
		plt.plot(timeList, regular(yhat))
		#plt.plot(peaks)
		k = 0
		for pt in peakTimeS:
			k += 1
			if k > 3:break
			plt.axvline(pt)
		print(period)
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



	def build_gram2():
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
		result = np.zeros((size_Y,size_X,len(timeList)))
		for x in range(size_X):
			print(x,size_X)
			for y in range(size_Y):
				result[y,x,:] = [None for _ in range(len(timeList))]
				lifegram[y,x] = None
				periodgram[y,x] = None
				if dataStorm[0][1][y,x]:
					rgbSeries = [(cybers[2][y,x][0]) for cybers in dataStorm]
					rgbSeries = monotonical(rgbSeries)
					rgbSeries = regular(rgbSeries)
					result[y,x] = rgbSeries
		return result

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
	minPeriod = 18
	maxPeriod = 28
	result = build_gram2()

	for tgb in range(100):
		img = result[:,:,tgb*20]
		img[0,0] = 0
		img[0,1] = 1
		image = cmap(img)
		for x in range(size_X):
			for y in range(size_Y):
				if not dataStorm[0][1][y,x] > 0:
					image[y,x] = [0,0,0,1]

		#plt.imsave(image)
		plt.imsave('poster/'+str(tgb*20)+'@'+str(timeList[tgb*20])+'.png',image)
		#plt.show()
		print ()
		#plt.savefig(str(tgb*20)+'@'+str(timeList[tgb*20])+'.png')
		#plt.clf()

	newRGBS = np.zeros((size_Y,size_X,len(timeList)))
	for y in range(size_Y):
		for x in range(size_X):
			pass






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
		#plt.imshow(lifegram)
		#plt.show()
		img = Image.fromarray(colorMat)
		img = img.resize((sub_x,sub_y))
		img = ImageTk.PhotoImage(img)

		canvas.create_image(0,0,image=img, anchor = 'nw')
		root.bind('<Button 1>', click)
		root.mainloop()





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