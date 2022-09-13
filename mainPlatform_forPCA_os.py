import calculationS as cal
import visualizationS as vis
from matplotlib import pyplot as plt
from matplotlib import pyplot

import math
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


plt.rcParams["figure.figsize"] = (12,6)

with open('data2.txt') as F:
	line = F.readline()
	headS = line.strip('\n').split('\t')
	N = len(headS)
	sampleS = [headS[n*3+1] for n in range(int(N/3))]
	geneS = []
	mainData = dict()
	for sample in sampleS:mainData[sample] = dict()
	while 1:
		line = F.readline()
		if not line:break
		linedata = line.strip('\n').split('\t')
		gene = linedata[0]
		geneS.append(gene)
		for sample_n in range(int(N/3)):
			sample = sampleS[sample_n]
			mainData[sample][gene] = [float(linedata[sample_n*3+1+k]) for k in range(3)]

def getData(sample, gene):
	tempSampleS = [sampleS[k] for k in sample2numS[sample]]
	data = [mainData[sample][gene] for sample in tempSampleS]
	return data

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


sample2numS = dict()
sample2numS['col_y'] = range(6)
sample2numS['ore_y'] = range(6,12)
sample2numS['col_o'] = range(12,18)
sample2numS['ore_o'] = range(18,24)
sample2color = dict()

sample2color['col_y'] = 'go'
sample2color['col_o'] = 'ro'
sample2color['ore_y'] = 'gx'
sample2color['ore_o'] = 'rx'

if 0:
	data_PCA = []
	sampleList = []
	import random
	for sample in ['col_y','col_o']:
		for t in range(6):
			for r in range(3):
				print(sample, t, r)
				temp = []
				for gene in geneS:
					tempData = getData(sample,gene)
					tempData = np.array(tempData)
					if 0:
						if (np.max(tempData)-np.min((tempData))) == 0:
							tempData = tempData * 0
						else:
							tempData = (tempData - np.mean(tempData))/np.std(tempData)
					temp.append((tempData[t][r]))
				data_PCA.append(temp)
				sampleList.append(sample)
	for sample in ['col_y','col_o']:
		for t in list(range(6))+[0]:
			for r in range(1):
				print(sample, t, r)
				temp = []
				for gene in geneS:
					tempData = getData(sample,gene)
					tempData = np.array(tempData)
					if 0:
						if (np.max(tempData)-np.min((tempData))) == 0:
							tempData = tempData * 0
						else:
							tempData = (tempData - np.mean(tempData))/np.std(tempData)
					temp.append(np.median(tempData[t]))
				data_PCA.append(temp)
				sampleList.append(sample)

	from sklearn import decomposition
	matrix_PCA = np.array(data_PCA)
	pca = decomposition.PCA(n_components=10)
	pca.fit(matrix_PCA)
	X = pca.transform(matrix_PCA)

	dump([X,sampleList,matrix_PCA],'PCAdat.dat')
X,sampleList = load('PCAdat.dat')

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#pcaPlot = fig.add_subplot(1,1,1)

print (sampleList)

#1,5,9,13,17,21

colorS = np.array([1,5,9,13,17,21])/24
colorS = [colorS[int(n/3)%6] for n in range(36)]
ZT = []

for n in range(6):
	plt.plot(X[18+n*3:18+n*3+3,0],X[18+n*3:18+n*3+3,1],'o',label='ZT_'+str(n*4+1))
plt.legend()
plt.show()

exit()

#for n in range(36):
ax.scatter([X[0:18,0]],[X[0:36,1]],[X[0:36,2]],c=colorS,cmap='viridis',marker='o',alpha=1)
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_zticks([])
#	print ([colorS[int(n/3)%6]])

#arrowplot(axes, x, y, z, narrs=30, dspace=0.5, direc='pos',hl=0.3, hw=6, c='black')
timer = ['1','5','9','13','17','21']
ct = 0
for n in range(36,42):
	a = Arrow3D(X[n:n+2,0],X[n:n+2,1],X[n:n+2,2], mutation_scale=20, lw=1, arrowstyle="-|>", color='g')
	ax.text(X[n,0],X[n,1],X[n,2],timer[ct],color='g',fontsize=20)
	ct+=1
	ax.add_artist(a)
ct = 0
if 0:
	for n in range(43,49):
		a = Arrow3D(X[n:n+2,0],X[n:n+2,1],X[n:n+2,2], mutation_scale=20, lw=1, arrowstyle="-|>", color='r')
		ax.text(X[n,0],X[n,1],X[n,2],timer[ct],color='r',fontsize=20)
		ct+=1
		ax.add_artist(a)
#arrowplot(ax,X[36:43,0],X[36:43,1],X[36:43,2],c='g')
#arrowplot(ax,X[43:50,0],X[43:50,1],X[43:50,2],c='r')
#ax.plot3D(X[36:43,0],X[36:43,1],X[36:43,2],'g>-')
#ax.plot3D(X[43:50,0],X[43:50,1],X[43:50,2],'r>-')
#plt.colorbar()
plt.show()

print (np.shape(X))



