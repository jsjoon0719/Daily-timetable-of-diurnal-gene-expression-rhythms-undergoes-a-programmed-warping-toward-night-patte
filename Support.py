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
from Useful import *




[TimeList, smoothCurve] = load('sample2')


def getPeriodFB2(Curve,TimeList):
	Curve2 = reviseLinear(Curve)
	[upPeakS, downPeakS] = findPeakThough(Curve2, TimeList)
	[firstPeak, though, secondPeak], [firstPeak_time, though_time, secondPeak_time] = getPeriodFB(upPeakS,downPeakS,TimeList)
	Varify(Curve, firstPeak, though, secondPeak, TimeList)
	plt.plot(TimeList,regular(Curve2))  
	for peak in upPeakS:
			plt.axvline(TimeList[peak],color='r')
	for peak in downPeakS:
			plt.axvline(TimeList[peak],color='b')




print (Process2PeriodFB(smoothCurve, TimeList, 1)) 

exit()
plt.subplot(3,1,1)
plt.plot(TimeList,smoothCurve)
plt.subplot(3,1,2)
getPeriodFB2(smoothCurve, TimeList)
plt.subplot(3,1,3)
diffSeries = smoothCurve[1:len(smoothCurve)]-smoothCurve[0:len(smoothCurve)-1]
getPeriodFB2(np.array(regular(savgol_filter(diffSeries, 51, 3) )), TimeList[1:])
plt.show()