import numpy as np
import random


def norm(series):
	temp = np.array(series)
	temp2 = (temp - np.min(temp))/(np.max(temp)-np.min(temp))
	return temp2

def getAtDt(series_normed, thre=0.5):
	N = len(series_normed)
	AtS = []
	DtS = []
	for n in range(N-1):
		if series_normed[n] <= thre and series_normed[n+1] > thre:AtS.append(n)
		if series_normed[n] > thre and series_normed[n+1] <= thre:DtS.append(n)
	if 1:
		if series_normed[N-1] <= thre and series_normed[0] > thre:AtS.append(N-1)
		if series_normed[N-1] > thre and series_normed[0] <= thre:DtS.append(N-1)
	return AtS, DtS

def cycling(series_normed, threS=[0.4,0.5,0.6]):
	testS = []
	for thre in threS:
		AtS, DtS = getAtDt(series_normed, thre)
		if len(AtS) == 1 and len(AtS) == 1:
			testS.append(1)
		else:
			testS.append(0)
	#return testS
	if len(testS) == sum(testS):
		return 1
	else:
		return 0

def getFWHM(series_normed, at, dt, thre=0.5):
	N = len(series_normed)
	result = []
	for t in [at, dt]:
		if t != N-1:
			v1, v2 = series_normed[t], series_normed[t+1]
			t1, t2 = t, t+1
		else:
			v1, v2 = series_normed[t], series_normed[0]
			t1, t2 = t, t+1
		result.append( ((thre-v1)*t2+(v2-thre)*t1)/(v2-v1) )
	a, d = result
	if a > d: a = a - N
	result = [a,d,d-a]
	return result


def testCycle(rawData, testN = 1000):
	youngData = rawData[0:18]
	oldData = rawData[18:36]
	fullResult = []
	for data in [youngData, oldData]:
		tempResult = [0, 0, [], [], []]#cycling, non-cycling, atS, dtS, FWHMs
		for _ in range(testN):
			temp = [random.choice(data[3*n:3*n+3]) for n in range(6)]
			series = norm(temp)
			if cycling(series):
				atS, dtS = getAtDt(series, 0.5)
				at, dt, FWHM = getFWHM(series, atS[0], dtS[0])
				tempResult[0] += 1
				tempResult[2].append(at*24/6+1)
				tempResult[3].append(dt*24/6+1)
				tempResult[4].append(FWHM*24/6)
			else:
				tempResult[1] += 1
		if tempResult[0] > 1:
			result = [1.0*tempResult[0]/testN]
			result.append([np.mean(tempResult[2]),np.std(tempResult[2])])#at
			result.append([np.mean(tempResult[3]),np.std(tempResult[3])])#dt
			result.append([np.mean(tempResult[4]),np.std(tempResult[4])])#FWHM
		else:
			result = [0, [0,0], [0,0], [0,0]]
		fullResult.append(result)
	return fullResult




		








			
