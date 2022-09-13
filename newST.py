from joblib import dump, load
import numpy as np
from sklearn import decomposition
from matplotlib import pyplot as plt
from CosinorPy import file_parser, cosinor, cosinor1
import pandas as pd
import os
import math
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn import tree

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PassiveAggressiveRegressor




with open('data.txt') as F:
	indexS = F.readline().strip('\n').split('\t')[1:]
	data = [[] for _ in indexS]
	print (indexS)
	geneS = []
	while 1:
		line = F.readline()
		if not line:break
		spline = line.strip('\n').split('\t')[1:]
		for n in range(len(spline)):
			data[n].append(float(spline[n]))
		geneS.append(line.split('\t')[0])

trainY = [1,1,1,5,5,5,9,9,9,13,13,13,17,17,17,21,21,21]
trainX = data[0:18]

def angle_between(p1, p2):
	    ang1 = np.arctan2(*p1[::-1])
	    ang2 = np.arctan2(*p2[::-1])
	    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def train_Model(trainX,trainY):
	N = 10
	angleRevisorS = [k/N*math.pi*2 for k in range(N)]
	#print (angleRevisorS)
	#angleRevisorS = [0]
	modelS_cos = [linear_model.Ridge() for _ in range(N)]
	modelS_sin = [linear_model.Ridge() for _ in range(N)]
	for k in range(N):
		angleRevisor = angleRevisorS[k]
		trainY_cos = [math.cos(angle/24*2*math.pi+angleRevisor) for angle in trainY]
		trainY_sin = [math.sin(angle/24*2*math.pi+angleRevisor) for angle in trainY]
		modelS_cos[k].fit(trainX,trainY_cos)
		modelS_sin[k].fit(trainX,trainY_sin)
	return modelS_cos, modelS_sin, angleRevisorS

def transform_Model(modelS_cos, modelS_sin, angleRevisorS, target):
	N = len(angleRevisorS)
	xS = []
	yS = []
	for n in range(N):
		pre_cos = modelS_cos[n].predict(target)[0]
		pre_sin = modelS_sin[n].predict(target)[0]
		pre_angle = angle_between((pre_cos,pre_sin),(1,0))*math.pi*2/360
		pre_angle = pre_angle - angleRevisorS[n]
		pre_x = math.cos(pre_angle)
		pre_y = math.sin(pre_angle)
		xS.append(pre_x)
		yS.append(pre_y)
	x = np.mean(xS)
	y = np.mean(yS)
	#print (x,y)
	result = angle_between((x,y),(1,0))*24/360
	return result


def getGene(g):
	young = []
	old = []
	for n in range(18):
		young.append(data[n][g])
		old.append(data[n+18][g])
	return young, old



trainY2 = trainY + [y+24 for y in trainY]


geneN = len(data[0])

df = file_parser.generate_test_data(phase = 0, n_components = 1, name="test1", noise=0.5, replicates = 3)
print (df)
#plt.plot(trainY2,getGene(4)[0]+getGene(4)[0])
#plt.show()
with open('cosinor.txt', 'w') as F:
	for g in range(geneN):
		d = {'test' : ['1' for _ in range(18)],'x': trainY, 'y': getGene(g)[0]}
		df = pd.DataFrame(d)
		df_results = cosinor.fit_group(df, period=24, plot=False)
		#df_best_fits = cosinor.get_best_fits(df_results, criterium='RSS', reverse = False)
		p1 = float(df_results['p'])
		d = {'test' : ['1' for _ in range(18)],'x': trainY, 'y': getGene(g)[1]}
		df = pd.DataFrame(d)
		df_results = cosinor.fit_group(df, period=24, plot=False)
		p2 = float(df_results['p'])
		print (p1, p2, g, geneS[g])
		F.write(str(p1)+'\t'+str(p2)+'\t'+str(g)+'\t'+geneS[g]+'\tX\n')



exit()



for t in range(6):
	new_trainX = []
	new_trainY = []
	new_testX = []
	for m in range(6):
		if t != m:
			for k in range(3):
				new_trainY.append(trainY[3*m+k])
				new_trainX.append(trainX[3*m+k])
		else:
			for k in range(3):
				new_testX.append(trainX[3*m+k])
	modelS_cos, modelS_sin, angleRevisorS = train_Model(new_trainX,new_trainY)
	for n in range(3):
		print(transform_Model(modelS_cos, modelS_sin, angleRevisorS, [new_testX[n]]),t)
