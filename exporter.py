import numpy as np
import funC as f
from matplotlib import pyplot as plt

geneS = []#['AT5G03910','AT4G01660','AT5G06980','AT5G01620','AT5G03910']
geneNameS = []
with open('targetS_GN.txt') as T:
	while 1:
		line = T.readline()
		if not line:break
		#continue
		gene, name = line.strip('\n').split(',')
		geneS.append(gene)
		geneNameS.append(name)
with open('result_final.txt', 'w') as R:
	for tn in range(len(geneS)):
		Target_gene = geneS[tn]
		Target_gene_name = geneNameS[tn]
		genename = Target_gene
		rawDb = dict()
		if 1:
			with open('data.txt') as F:
				print (F.readline())
				while 1:
					line = F.readline()
					
					if not line:break
					spline = line.strip('\n').split('\t')
					gene = spline[0]
					if gene != Target_gene:continue
					datS = [float(dat) for dat in spline[1:]]
					#datS = f.norm(datS)
					temp = [np.mean(datS[3*n:3*n+3]) for n in range(12)]
					rawDb[gene] = [datS[:], f.norm(temp[0:6]), f.norm(temp[6:12])]
					lowS = [np.min(datS[3*n:3*n+3]) for n in range(12)]
					highS = [np.max(datS[3*n:3*n+3]) for n in range(12)]
					stdS = [np.std(datS[3*n:3*n+3]) for n in range(12)]
					if 1:
						R.write(genename+'\t'+Target_gene_name+'\n')
						for n in range(6):
							R.write(str(temp[n])+'\t')
						for n in range(6):
							R.write(str(temp[n])+'\t')
						R.write('\n')
						for n in range(6):
							R.write(str(stdS[n])+'\t')
						for n in range(6):
							R.write(str(stdS[n])+'\t')
						R.write('\n')
						for n in range(6,12):
							R.write(str(temp[n])+'\t')
						for n in range(6,12):
							R.write(str(temp[n])+'\t')
						R.write('\n')
						for n in range(6,12):
							R.write(str(stdS[n])+'\t')
						for n in range(6,12):
							R.write(str(stdS[n])+'\t')
						R.write('\n')