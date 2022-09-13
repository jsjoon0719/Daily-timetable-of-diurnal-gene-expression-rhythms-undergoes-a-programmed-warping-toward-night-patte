from scipy.signal import savgol_filter
import numpy as np
from joblib import dump, load
import tkinter as tk
from sklearn.svm import SVC
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import sys
from scipy.signal import find_peaks
from joblib import dump, load
import os
from sklearn.linear_model import LinearRegression



def Process2PeriodFB(curve, TimeList, drawPlots):
	smoothCurve = smoothen(curve)
	revised = reviseLinear(smoothCurve)
	[upPeakS, downPeakS] = findPeakThough(revised, TimeList)
	temp = getPeriodFB(upPeakS,downPeakS,TimeList)
	#print (temp)
	if temp == 'NoResult':
		if drawPlots:
			ax1 = plt.subplot(2,1,1)
			ax1.title.set_text('RawData')
			plt.plot(TimeList, curve)
			plt.yticks([])
			ax2 = plt.subplot(2,1,2)
			plt.plot(TimeList,revised)
			plt.yticks([])
			ax2.title.set_text('Revised')
			plt.yticks([])
			plt.tight_layout()
			plt.show()




		return 0
	diffCurve = smoothen(derivative(curve))
	[firstPeak, though, secondPeak], [firstPeak_time, though_time, secondPeak_time] = temp
	ver = Varify(diffCurve, firstPeak, though, secondPeak, TimeList)
	if drawPlots:	print ("draw?", drawPlots, ver)
	if not ver:
		if drawPlots:
			ax1 = plt.subplot(3,1,1)
			ax1.title.set_text('RawData')
			plt.axvline(TimeList[firstPeak],color ='r')
			plt.axvline(TimeList[though],color ='g')
			plt.axvline(TimeList[secondPeak],color ='b')
			plt.plot(TimeList, curve)
			plt.yticks([])
			ax2 = plt.subplot(3,1,2)
			plt.axvline(TimeList[firstPeak],color ='r')
			plt.axvline(TimeList[though],color ='g')
			plt.axvline(TimeList[secondPeak],color ='b')
			plt.plot(TimeList,revised)
			plt.yticks([])
			ax2.title.set_text('Revised')
			ax3 = plt.subplot(3,1,3)
			plt.axvline(TimeList[firstPeak],color ='r')
			plt.axvline(TimeList[though],color ='g')
			plt.axvline(TimeList[secondPeak],color ='b')
			ax3.title.set_text('Differentiated')
			plt.plot(TimeList,diffCurve)
			plt.yticks([])
			plt.tight_layout()
			plt.show()





		#return 0
	if ver:	[firstInf, secondInf] = ver
	if drawPlots and ver:
		ax1 = plt.subplot(3,1,1)
		ax1.title.set_text('RawData')
		plt.axvline(TimeList[firstPeak],color ='r')
		plt.axvline(TimeList[though],color ='g')
		plt.axvline(TimeList[secondPeak],color ='b')
		plt.axvline(TimeList[firstInf],color ='black')
		plt.axvline(TimeList[secondInf],color ='black')
		plt.plot(TimeList, curve)
		plt.yticks([])
		ax2 = plt.subplot(3,1,2)
		plt.axvline(TimeList[firstPeak],color ='r')
		plt.axvline(TimeList[though],color ='g')
		plt.axvline(TimeList[secondPeak],color ='b')
		plt.axvline(TimeList[firstInf],color ='black')
		plt.axvline(TimeList[secondInf],color ='black')
		plt.plot(TimeList,revised)
		plt.yticks([])
		ax2.title.set_text('Revised')
		ax3 = plt.subplot(3,1,3)
		plt.axvline(TimeList[firstPeak],color ='r')
		plt.axvline(TimeList[though],color ='g')
		plt.axvline(TimeList[secondPeak],color ='b')
		plt.axvline(TimeList[firstInf],color ='black')
		plt.axvline(TimeList[secondInf],color ='black')
		ax3.title.set_text('Differentiated')
		plt.plot(TimeList,diffCurve)
		plt.yticks([])
		plt.tight_layout()
		plt.show()





	return though_time - firstPeak_time, secondPeak_time - though_time


def Process2PeriodFB2(curve, TimeList, drawPlots):
	print('pp')
	smoothCurve = smoothen(curve)
	revised = reviseLinear(smoothCurve)
	[upPeakS, downPeakS] = findPeakThough(revised, TimeList)
	temp = getPeriodFB(upPeakS,downPeakS,TimeList)
	#print (temp)
	if temp == 'NoResult':
		if drawPlots:
			ax1 = plt.subplot(2,1,1)
			ax1.title.set_text('RawData')
			plt.plot(TimeList, curve)
			plt.yticks([])
			ax2 = plt.subplot(2,1,2)
			plt.plot(TimeList,revised)
			plt.yticks([])
			ax2.title.set_text('Revised')
			plt.yticks([])
			plt.tight_layout()
			plt.savefig('temp.png')




		return 0
	diffCurve = smoothen(derivative(curve))
	[firstPeak, though, secondPeak], [firstPeak_time, though_time, secondPeak_time] = temp
	ver = Varify(diffCurve, firstPeak, though, secondPeak, TimeList)
	if drawPlots:	print ("draw?", drawPlots, ver)
	if not ver:
		if drawPlots:
			ax1 = plt.subplot(3,1,1)
			ax1.title.set_text('RawData')
			plt.axvline(TimeList[firstPeak],color ='r')
			plt.axvline(TimeList[though],color ='g')
			plt.axvline(TimeList[secondPeak],color ='b')
			plt.plot(TimeList, curve)
			plt.yticks([])
			ax2 = plt.subplot(3,1,2)
			plt.axvline(TimeList[firstPeak],color ='r')
			plt.axvline(TimeList[though],color ='g')
			plt.axvline(TimeList[secondPeak],color ='b')
			plt.plot(TimeList,revised)
			plt.yticks([])
			ax2.title.set_text('Revised')
			ax3 = plt.subplot(3,1,3)
			plt.axvline(TimeList[firstPeak],color ='r')
			plt.axvline(TimeList[though],color ='g')
			plt.axvline(TimeList[secondPeak],color ='b')
			ax3.title.set_text('Differentiated')
			plt.plot(TimeList,diffCurve)
			plt.yticks([])
			plt.tight_layout()
			plt.savefig('temp.png')





		#return 0
	if ver:	[firstInf, secondInf] = ver
	if drawPlots and ver:
		ax1 = plt.subplot(3,1,1)
		ax1.title.set_text('RawData')
		plt.axvline(TimeList[firstPeak],color ='r')
		plt.axvline(TimeList[though],color ='g')
		plt.axvline(TimeList[secondPeak],color ='b')
		plt.axvline(TimeList[firstInf],color ='black')
		plt.axvline(TimeList[secondInf],color ='black')
		plt.plot(TimeList, curve)
		plt.yticks([])
		ax2 = plt.subplot(3,1,2)
		plt.axvline(TimeList[firstPeak],color ='r')
		plt.axvline(TimeList[though],color ='g')
		plt.axvline(TimeList[secondPeak],color ='b')
		plt.axvline(TimeList[firstInf],color ='black')
		plt.axvline(TimeList[secondInf],color ='black')
		plt.plot(TimeList,revised)
		plt.yticks([])
		ax2.title.set_text('Revised')
		ax3 = plt.subplot(3,1,3)
		plt.axvline(TimeList[firstPeak],color ='r')
		plt.axvline(TimeList[though],color ='g')
		plt.axvline(TimeList[secondPeak],color ='b')
		plt.axvline(TimeList[firstInf],color ='black')
		plt.axvline(TimeList[secondInf],color ='black')
		ax3.title.set_text('Differentiated')
		plt.plot(TimeList,diffCurve)
		plt.yticks([])
		plt.tight_layout()
		plt.savefig('temp.png')





	return though_time - firstPeak_time, secondPeak_time - though_time
def getPeriodFB(upPeakS,downPeakS,TimeList, minTime = 10, minGap = 4):
	firstPeak = 0
	firstPeak_time = 0
	though = 0
	though_time = 0
	secondPeak = 0
	secondPeak_time = 0
	for peak in upPeakS:
		if TimeList[peak] > minTime:
			firstPeak = peak
			firstPeak_time = TimeList[peak]
			break
	for peak in downPeakS:
		if TimeList[peak] > firstPeak_time + minGap:
			though = peak
			though_time = TimeList[peak]
			break
	for peak in upPeakS:
		if TimeList[peak] > though_time + minGap:
			secondPeak = peak
			secondPeak_time = TimeList[peak]
	if not (firstPeak_time and though_time and secondPeak_time):
		return 'NoResult'
	return [firstPeak, though, secondPeak], [firstPeak_time, though_time, secondPeak_time]

def Varify(diffCurve, firstPeak, though, secondPeak, TimeList):
	#print (diffCurve)
	upPeakS, downPeakS = findPeakThough(diffCurve, TimeList)
	n1 = CountBetween(firstPeak,though,upPeakS)
	n2 = CountBetween(though,secondPeak,upPeakS)
	n3 = CountBetween(firstPeak,though,downPeakS)
	n4 = CountBetween(though,secondPeak,downPeakS)
	#print (n1, n2, n3, n4)
	if n1 == 0 and n2 == 1 and n3 == 1 and n4 == 0:
		first = GiveBetween(though,secondPeak,upPeakS)
		second = GiveBetween(firstPeak,though,downPeakS)
		return [first,second]
	else:
		return 0


def CountBetween(A,B,series):
	result = 0
	for s in series:
		if s > A and s < B:
			result += 1
	return result

def GiveBetween(A,B,series):
	result = 0
	for s in series:
		if s > A and s < B:
			return s




def regular(series):
	maxer = max(series)
	miner = min(series)
	result = [1.0*(ser-miner)/(maxer-miner) for ser in series]
	return result
def smoothen(Curve):return savgol_filter(Curve, 51, 3)
def derivative(Curve):
	temp = list(Curve[1:len(Curve)]-Curve[0:len(Curve)-1])
	temp.append(temp[-1])
	return temp
def findPeakThough(Curve, TimeList, gap_required = 12):
	timeGap = TimeList[1]-TimeList[0]
	gap_needed = int(15/timeGap)
	peaks1, properties = find_peaks(Curve,distance = gap_needed)
	peaks2, properties = find_peaks(-Curve,distance = gap_needed)
	peaks = list(set(list(peaks1)+list(peaks2)))
	peaks.sort()
	upPeakS = []
	downPeakS = []
	periodS = []
	peakTimeS = []
	for peak in peaks:
		time = TimeList[peak]
		peakTimeS.append(time)
		yValue = Curve[peak]
		try:
			yAverage = np.mean(Curve[peak-2:peak+3])
		except:
			continue
		if yValue > yAverage:
			upPeakS.append(peak)
		else:
			downPeakS.append(peak)
	return upPeakS, downPeakS
def reviseLinear(Curve):
	psudoX = [[n] for n in range(len(Curve))]
	psudoY = Curve
	reg = LinearRegression().fit(psudoX,psudoY)
	reged = reg.predict(psudoX)
	result = Curve- reged
	return result
