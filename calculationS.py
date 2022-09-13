from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import visualizationS as vis
from scipy.stats import norm

def revision(data1, data2):
	N = len(data1)
	thetaS = [math.pi*2/N*k for k in range(N)]
	iScale = np.mean(data1)/np.mean(data2)
	iST_start = 0
	iST_gapS = [1 for _ in data1]
	paramS = [iScale, iST_start] + iST_gapS
	res = minimize(getFit, x0=paramS, args=(data1,data2), method='nelder-mead', options={'maxiter':1000, 'disp':False})
	scale,ST_start,ST_gapS = abs(res.x[0]), res.x[1], res.x[2:]
	thetaS2_revised = gap2angleS(ST_start,ST_gapS)
	return scale, thetaS2_revised

def gap2angleS(start, gapS):
	gapS = abs(np.array(gapS))
	gapS = gapS/sum(gapS)
	temp = []
	for g in gapS:
		if g > 0.45:temp.append(0.45)
		if g < 0.05:temp.append(0.05)
		else:temp.append(g)
	temp = 2*math.pi*np.array(temp)/sum(temp)
	angle2 = [start]
	for gap in temp:angle2.append(angle2[-1]+gap)
	return angle2

def getFit(paramS,data1,data2):
	scale,ST_start,ST_gapS = abs(paramS[0]), paramS[1], paramS[2:]
	N = len(data1)
	data2 = np.array(data2)*scale
	thetaS1 = [math.pi*2/N*k for k in range(N)]
	thetaS2 = gap2angleS(ST_start,ST_gapS)
	try:score, _ = getScore(data1,data2,thetaS1,thetaS2)
	except:score = 0
	return -score

def sortPool(pool):
	tempDict = dict()
	for [t1,d] in pool:
		t = t1 % (math.pi*2)
		if t in tempDict.keys():tempDict[t].append(d)
		else:tempDict[t] = [d]
	thetaS = list(tempDict.keys())
	thetaS.sort()
	return thetaS, [tempDict[t] for t in thetaS]



def getPvalue(data1, data2, thetaS1=None, thetaS2=None, statN=400):
	N = len(data1)
	if not thetaS1:thetaS1 = [math.pi*2/N*k for k in range(N)]
	if not thetaS2:thetaS2 = [math.pi*2/N*k for k in range(N)]
	TotalPool = []
	for n in range(N):
		t1, t2 = thetaS1[n], thetaS2[n]
		for dat in data1[n]:TotalPool.append([t1,dat])
		for dat in data2[n]:TotalPool.append([t2,dat])


	score, _ = getScore(data1, data2, thetaS1, thetaS2)
	dist = [0 for _ in range(statN)]
	for s in range(statN):
		pool1 = random.sample(TotalPool,int(len(TotalPool)/2))
		pool2 = random.sample(TotalPool,int(len(TotalPool)/2))
		#pool2 = []
		#for dat in TotalPool:
	#		if not dat in pool1:pool2.append(dat)
		tempThetaS1, tempData1 = sortPool(pool1)
		tempThetaS2, tempData2 = sortPool(pool2)
		#print (tempData1)
		tempscore, _ = getScore(tempData1, tempData2, tempThetaS1, tempThetaS2)
		dist[s] = tempscore
	m = np.mean(dist)
	s = np.std(dist)
	pValue = norm.cdf((score-m)/s)
	return pValue, dist

def getScore(data1, data2, thetaS1=None, thetaS2=None): 
	p1, p2 = genPolygon_fromData(data1,thetaS1), genPolygon_fromData(data2,thetaS2)
	score, inter = getRatio(p1,p2)
	return score, inter

def genPolygon_fromData(data,thetaS=None,divN=10):
	N = len(data)
	if not thetaS:thetaS = [math.pi*2/N*k for k in range(N)]
	radiusS = [np.mean(dat) for dat in data]
	return genPolygon(radiusS, thetaS,divN)

def genPolygon(radiusS,thetaS=None,divN=10):
	if not thetaS:thetaS = [math.pi*2/N*k for k in range(N)]
	N = len(radiusS)
	tempRadiusS = list(radiusS) + [radiusS[0]]
	temp = [dat for dat in tempRadiusS]
	temp.sort()
	for dat in temp:
		if dat:
			nonzero = dat/100
			break
	if not max(temp):nonzero = 1/100000

	tempThetaS = list(thetaS) + [math.pi*2+thetaS[0]]
	cordiS = []
	for n in range(N):
		r1, r2 = tempRadiusS[n], tempRadiusS[n+1]
		if not r1:r1=nonzero
		if not r2:r2=nonzero	
		if 1:
			t1, t2 = tempThetaS[n], tempThetaS[n+1]
			for k in range(divN):
				r = r1 + (r2-r1)*k/divN
				t = t1 + (t2-t1)*k/divN
				x, y = math.cos(t)*r, math.sin(t)*r
				cordiS.append((x,y))
	return Polygon(cordiS+[cordiS[0]])

def getRatio(p1,p2):
	inter = p1.intersection(p2)

	a, b, c = p1.area, p2.area, inter.area
	return (c/(a+b-c)), inter



#data1 = [[97.5692893, 93.38422566, 91.98199352], [82.953878, 83.01797436, 92.76574815], [103.3835033, 112.2717063, 96.550897], [102.265661, 97.31569335, 95.17615587], [90.98050256, 103.7210887, 110.5636429], [101.8407648, 103.7998584, 102.036711]]
#data2 = [[101.5220167, 109.6062848, 99.91746937], [135.7543631, 112.7390138, 118.6809006], [127.0596303, 101.9830672, 139.5955912], [109.4793347, 122.0368104, 107.0324288], [109.0273809, 97.20891164, 117.7343093], [118.5163638, 125.3012653, 125.5690783]]