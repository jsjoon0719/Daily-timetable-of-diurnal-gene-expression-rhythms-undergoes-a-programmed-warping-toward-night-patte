import numpy as np
from PIL import Image, ImageTk, ImageEnhance

def resizeArray(array_2D, wantX, wantY):
	[size_Y,size_X] = np.shape(array_2D)
	scaleFactor = max(wantX/size_X, wantY/size_Y)
	newimg = Image.fromarray(array_2D.astype('uint8'))
	newimg = newimg.resize((int(size_X*scaleFactor),int(size_Y*scaleFactor)))
	return np.array(newimg)

def resizeArray2(array_2D, wantX, wantY):
	[size_Y,size_X] = np.shape(array_2D)
	scaleFactor = max(wantX/size_X, wantY/size_Y)
	newimg = Image.fromarray(array_2D.astype('uint8'))
	#newimg = newimg.resize((int(size_X*scaleFactor),int(size_Y*scaleFactor)))
	return np.array(newimg)

def regularizeCurve(series, timeSeries, gap = 12):
	result = []
	N = len(series)
	for n in range(N):
		y = series[n] 
		t = timeSeries[n]
		temp = []
		for m in range(N):
			y2 = series[m]
			t2 = timeSeries[m]
			if abs(t2-t) < gap:temp.append(y2)
		maxy = max(temp)
		miny = min(temp)
		newy = (y-miny)/(maxy-miny)
		result.append(newy)
	return result



def regularizeCurve2(series, timeSeries, gap = 12):
	result = []
	N = len(series)
	for n in range(N):
		y = series[n]
		t = timeSeries[n]
		temp = []
		for m in range(N):
			y2 = series[m]
			t2 = timeSeries[m]
			if abs(t2-t) < gap:temp.append(y2)
		maxy = max(temp)
		miny = min(temp)
		newy = (y-miny)/(maxy-miny)
		result.append(newy)
	newResult = []
	pre = 0
	post = 0
	pre_dark=0
	post_dark=0
	if 0:
		for n in range(N-1):
			if timeSeries[n+1]%24 > 8 and timeSeries[n]%24 < 8:
				pre = result[n]
				post = result[n+1]
			if timeSeries[n]%24 > 8 or timeSeries[n]%24 < 0:
				newResult.append(result[n]-post+pre)
			else:
				newResult.append(result[n]-post_dark+pre_dark)
			if timeSeries[n+1]%24 < 8 and timeSeries[n]%24 > 8:
				pre_dark = newResult[n]
				post_dark = result[n+1]

	newResult.append(result[-1])


	return regularizeCurve(result,timeSeries)