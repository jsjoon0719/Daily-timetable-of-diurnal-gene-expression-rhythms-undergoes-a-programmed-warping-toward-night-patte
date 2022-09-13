import numpy as np
import funC as f
from matplotlib import pyplot as plt

rawDb = dict()
count = 0
with open('result2.txt','w') as R:
	R.write('Gene\tP_value\tAscendingT\tAscendingT_std\tDescendingT\tDescendingT_std\tFWHM\tFWHM_std\t')
	R.write('P_value\tAscendingT\tAscendingT_std\tDescendingT\tDescendingT_std\tFWHM\tFWHM_std\tX\n')
	with open('data.txt') as F:
		print (F.readline())
		while 1:
			#if count > 10:break
			if count % 100 == 0:
				print (count, 'genes complete')
			count += 1
			line = F.readline()
			if not line:break
			spline = line.strip('\n').split('\t')
			gene = spline[0]
			if gene != 'AT5G16410':continue
			R.write(gene+'\t')
			datS = [float(dat) for dat in spline[1:]]
			temp = [np.mean(datS[3*n:3*n+3]) for n in range(12)]
			rawDb[gene] = [datS[:], f.norm(temp[0:6]), f.norm(temp[6:12])]
			plt.title(gene)
			plt.plot([1+4*n for n in range(12)],temp[0:6]+temp[0:6], label='21 DAS')
			plt.plot([1+4*n for n in range(12)],temp[6:12]+temp[6:12], label='35 DAS')
			plt.legend()
			plt.xlabel('ZT')
			plt.ylabel('Expression')
			plt.show()
			[p,[at,atstd],[dt,dtstd],[fwhm,fwhmstd]] = (f.testCycle(rawDb[gene][0]))[0]
			p1 = p
			for dat in [p, at, atstd, dt, dtstd, fwhm, fwhmstd]:R.write(str(dat)+'\t')
			[p,[at,atstd],[dt,dtstd],[fwhm,fwhmstd]] = (f.testCycle(rawDb[gene][0]))[1]
			p2 = p
			for dat in [p, at, atstd, dt, dtstd, fwhm, fwhmstd]:R.write(str(dat)+'\t')
			if p1 == 1 and p2 == 0:
				print ('gene')
			R.write('\n')
			
