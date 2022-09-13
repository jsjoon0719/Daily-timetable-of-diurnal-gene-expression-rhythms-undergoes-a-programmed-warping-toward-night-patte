import numpy as np
import funC as f
from matplotlib import pyplot as plt

Target_gene = 'AT5G17490'
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
			with open(genename+'.txt', 'w') as R:
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
				
			#print(f.testCycle(rawDb[gene][0])[0], 'youngCycle')
			#print(f.testCycle(rawDb[gene][0])[1], 'OldCycle')
			print (lowS)
			print (highS)
			plt.title(gene)
			plt.fill_between([1+4*n for n in range(12)],lowS[0:6]+lowS[0:6],highS[0:6]+highS[0:6],facecolor='g', alpha=0.5)
			plt.fill_between([1+4*n for n in range(12)],lowS[6:12]+lowS[6:12],highS[6:12]+highS[6:12],facecolor='r', alpha=0.5)
			plt.plot([1+4*n for n in range(12)],temp[0:6]+temp[0:6], 'go-', label='21 DAS')
			plt.plot([1+4*n for n in range(12)],temp[6:12]+temp[6:12], 'ro-', label='35 DAS')
			plt.legend()
			plt.xlabel('ZT')
			plt.ylabel('Expression')
			y0,ym = (plt.ylim())
			plt.clf()
			#plt.subplot(121)
			plt.fill_betweenx([y0,ym],0,16, color='yellow', alpha=.2, linewidth=0)
			plt.fill_betweenx([y0,ym],16,24, color='gray', alpha=.2, linewidth=0)
			plt.fill_betweenx([y0,ym],24,40, color='yellow', alpha=.2, linewidth=0)
			plt.fill_betweenx([y0,ym],40,48, color='gray', alpha=.2, linewidth=0)
			
			plt.title(genename)
			plt.fill_between([1+4*n for n in range(12)],lowS[0:6]+lowS[0:6],highS[0:6]+highS[0:6],facecolor='g', alpha=0.5)
			plt.fill_between([1+4*n for n in range(12)],lowS[6:12]+lowS[6:12],highS[6:12]+highS[6:12],facecolor='r', alpha=0.5)
			plt.plot([1+4*n for n in range(12)],temp[0:6]+temp[0:6], 'go-', label='21 DAS')
			plt.plot([1+4*n for n in range(12)],temp[6:12]+temp[6:12], 'ro-', label='35 DAS')
			plt.legend()
			plt.xlabel('ZT')
			plt.ylabel('Expression')
			y0,ym = (plt.ylim())
			plt.show()

			datS =  list(f.norm(datS[0:18]))+list(f.norm(datS[18:36]))
			temp = [np.mean(datS[3*n:3*n+3]) for n in range(12)]
			rawDb[gene] = [datS[:], f.norm(temp[0:6]), f.norm(temp[6:12])]
			lowS = [np.min(datS[3*n:3*n+3]) for n in range(12)]
			highS = [np.max(datS[3*n:3*n+3]) for n in range(12)]

			plt.title('MYC4')
			plt.fill_between([1+4*n for n in range(12)],lowS[0:6]+lowS[0:6],highS[0:6]+highS[0:6],facecolor='g', alpha=0.5)
			plt.fill_between([1+4*n for n in range(12)],lowS[6:12]+lowS[6:12],highS[6:12]+highS[6:12],facecolor='r', alpha=0.5)
			plt.plot([1+4*n for n in range(12)],temp[0:6]+temp[0:6], 'go-', label='21 DAS')
			plt.plot([1+4*n for n in range(12)],temp[6:12]+temp[6:12], 'ro-', label='35 DAS')
			plt.legend()
			plt.xlabel('ZT')
			plt.ylabel('Normalized Expression')
			y0,ym = (plt.ylim())
			plt.clf()
			#plt.subplot(121)
			plt.fill_betweenx([y0,ym],0,16, color='yellow', alpha=.2, linewidth=0)
			plt.fill_betweenx([y0,ym],16,24, color='gray', alpha=.2, linewidth=0)
			plt.fill_betweenx([y0,ym],24,40, color='yellow', alpha=.2, linewidth=0)
			plt.fill_betweenx([y0,ym],40,48, color='gray', alpha=.2, linewidth=0)
			
			plt.title('MYC2')
			plt.fill_between([1+4*n for n in range(12)],lowS[0:6]+lowS[0:6],highS[0:6]+highS[0:6],facecolor='g', alpha=0.5)
			plt.fill_between([1+4*n for n in range(12)],lowS[6:12]+lowS[6:12],highS[6:12]+highS[6:12],facecolor='r', alpha=0.5)
			plt.plot([1+4*n for n in range(12)],temp[0:6]+temp[0:6], 'go-', label='21 DAS')
			plt.plot([1+4*n for n in range(12)],1.2*np.array(temp[6:12]+temp[6:12]), 'ro-', label='35 DAS')
			plt.legend()
			plt.xlabel('ZT')
			plt.ylabel('Normalized Expression')
			y0,ym = (plt.ylim())
			plt.show()
			[p,[at,atstd],[dt,dtstd],[fwhm,fwhmstd]] = (f.testCycle(rawDb[gene][0]))[0]
			p1 = p
			[p,[at,atstd],[dt,dtstd],[fwhm,fwhmstd]] = (f.testCycle(rawDb[gene][0]))[1]
			p2 = p
