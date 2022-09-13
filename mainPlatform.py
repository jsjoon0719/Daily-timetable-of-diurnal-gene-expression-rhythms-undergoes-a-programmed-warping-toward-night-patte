import calculationS as cal
import visualizationS as vis
from matplotlib import pyplot as plt
import math
import numpy as np
import os



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

sample2numS = dict()
sample2numS['col_y'] = range(6)
sample2numS['ore_y'] = range(6,12)
sample2numS['col_o'] = range(12,18)
sample2numS['ore_o'] = range(18,24)


def sorter(series):
	temp = []
	for n in range(6):
		temp.append([series[n],n])
	start = max(temp)[1]
	result = []
	for m in range(6):
		result.append(series[(start+m)%6])
	return regularize(result)

def regularize(series):
	temp = np.array(series)
	if (np.max(series)-np.min(series)) == 0:1/0
	result = (temp-np.min(series))/(np.max(series)-np.min(series))
	return result




gene = geneS[195]
sample1 = 'col_y'
sample2 = 'col_o'
codeName=  'result_'+sample1+'_vs_'+sample2

fig = plt.figure()
try:
	os.makedirs(codeName)
except:
	pass

from sklearn.cluster import KMeans

def compare(gene):
	youngSeries = (np.median(getData('col_y',gene),axis=1))
	oldSereis = (np.median(getData('col_o',gene),axis=1))
	sYoung = sorter(youngSeries)
	sOld = sorter(oldSereis)
	return sYoung,sOld

youngDB = []
oldDB = []
vGeneS = []
for gene in geneS:
	try:
		sYoung, sOld = compare(gene)
	except:
		pass
	youngDB.append(sYoung)
	oldDB.append(sOld)
	vGeneS.append(gene)
kmeans = KMeans(n_clusters=3, random_state=0).fit(youngDB)
youngCluster = kmeans.predict(youngDB)
oldCluster = kmeans.predict(oldDB)
colorS = ['r','g','b','k','orange']
with open('cluster3.txt','w') as F:
	for n in range(len(youngCluster)):
		F.write(vGeneS[n]+'\t'+str(youngCluster[n])+'\t'+str(oldCluster[n])+'\n')
import random
for m in range(3):
	plt.subplot(3,1,m+1)
	temp = np.zeros(6)
	tempn = 0
	for n in range(len(youngCluster)):
		if youngCluster[n] != m:continue
		temp += youngDB[n]
		tempn += 1
		if random.random()<0.99:continue
		#c = colorS[youngCluster[n]]
		plt.plot(list(youngDB[n][1:6])+list(youngDB[n][0:5]),color='gray',linewidth=1)
	temp = temp/tempn
	plt.plot(list(temp)[1:6]+list(temp)[0:5],color=colorS[m],linewidth=5)
plt.show()


exit()

with open(codeName+'2www_.txt', 'w') as F:
	for gn in range(1):
		gene = geneS[gn]
		gene = 'AT1G80820'
		plt.close(fig)
		try:
			print (gn, len(geneS), sample1, sample2)
			plt.clf()
			fig = plt.figure()
			rawDataPlot = fig.add_subplot(2,3,1)
			rawPolarPlot = fig.add_subplot(2,3,2)
			phasePlot = fig.add_subplot(2,3,3)
			revisedDataPlot = fig.add_subplot(2,3,4)
			revisedPolarPlot = fig.add_subplot(2,3,5)
			infoPlot = fig.add_subplot(2,3,6)
			result = compareTwo(gene,sample1,sample2,visual=1)
			for res in result:
				F.write(str(res)+'\t')
			F.write('X\n')
		except:
			print (gene+' Failed!')
