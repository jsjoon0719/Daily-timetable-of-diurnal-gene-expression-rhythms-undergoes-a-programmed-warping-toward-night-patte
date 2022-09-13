import numpy as np
import os

fileNameS = os.listdir('ResultS')
dataBase = dict()
for fileName in fileNameS:
	dataBase[fileName] = dict()
	with open('ResultS/'+fileName) as F:
		while 1:
			line = F.readline()
			if not line:break
			spline = line.strip('\n').split('\t')
			gene = spline[0]
			essentialS = spline[1:7]
			nonS = spline[1:-1]
			dataBase[fileName][gene] = [essentialS,nonS]

geneS = dataBase[fileName].keys()
geneS = set(geneS)
for fileName in fileNameS:
	temp = dataBase[fileName].keys()
	temp = set(temp)
	geneS = geneS.intersection(temp)

geneS = list(geneS)
geneS.sort()

with open('Sorted.txt', 'w') as R:
	R.write('\t')
	for fileName in fileNameS:
		for _ in range(6):R.write(fileName.split('result_')[1].split('_.')[0]+'\t')
	for fileName in fileNameS:
		for _ in range(10):R.write(fileName.split('result_')[1].split('_.')[0]+'\t')
	R.write('X\n')
	R.write('\t')
	for fileName in fileNameS:
		R.write('score\tp\ts_r\tp_r\tFC\tST\t')
	for fileName in fileNameS:
		R.write('morning\tevening\tnight\t1\t5\t9\t13\t17\t21\t25\t')
	R.write('X\n')

	for gene in geneS:
		R.write(gene + '\t')
		for fileName in fileNameS:
			for dat in dataBase[fileName][gene][0]:R.write(dat+'\t')
		for fileName in fileNameS:
			for dat in dataBase[fileName][gene][1]:R.write(dat+'\t')
		R.write('X\n')


