with open('DATA.txt') as F:
	F.readline()
	database = dict()
	geneS = []
	while 1:
		line = F.readline()
		if not line:break
		spline = line.strip('\n').split('\t')
		gene = spline[0]
		YA, YD, OA, OD = spline[1:5]
		YA, YD, OA, OD = float(YA), float(YD), float(OA), float(OD)
		geneS.append(gene)
		database[gene] = [YA, YD, OA, OD]

with open('GO_terms.txt') as F:
	F.readline()
	goDB = dict()
	while 1:
		line = F.readline()
		if not line:break
		spline = line.strip('\n').split('\t')
		go = spline[1].split('~')[1]
		genes = spline[5].split(', ')
		goDB[go] = genes

print (goDB)
from matplotlib import pyplot as plt

def BG(geneList):
	N = len(geneList)
	newList = []
	temp = []
	master_y = [0 for _ in range(48)]
	master_o = [0 for _ in range(48)]
	for n in range(N):
		gene = geneList[n]
		YA, YD, OA, OD = database[gene]
		YF = YD-YA
		OF = OD-OA
		if abs(OF-YF) < 2:continue
		for m in range(48):
			if m/2 > YA and m/2 < YD:
				master_y[m] += 1
			if m/2 > OA and m/2 < OD:
				master_o[m] += 1
			if m/2+24 > YA and m/2+24 < YD:
				master_y[m] += 1
			if m/2+24 > OA and m/2+24 < OD:
				master_o[m] += 1
		YF = YD-YA
		OF = OD-OA
		if (OF-YF) < 1:continue
		temp.append([YA+YD,YA-YD,OD-OA,gene])

	temp.sort()
	for [_,_,_,gene] in temp:
		newList.append(gene)
	N = len(newList)
	for n in range(N):
		gene = newList[n]
		YA, YD, OA, OD = database[gene]
		YF = YD-YA
		OF = OD-OA
		plt.subplot(3,2,1)
		plt.plot([YA,YD],[1.0*n/N,1.0*n/N],'g',linewidth=100/N)
		plt.subplot(3,2,2)
		plt.plot([OA,OD],[1.0*n/N,1.0*n/N],'r',linewidth=100/N)

	N = len(geneList)
	newList = []
	temp = []
	for n in range(N):
		gene = geneList[n]
		YA, YD, OA, OD = database[gene]
		YF = YD-YA
		OF = OD-OA
		if (OF-YF) > -1:continue
		temp.append([YA+YD,YA-YD,OD-OA,gene])

	temp.sort()
	for [_,_,_,gene] in temp:
		newList.append(gene)
	N = len(newList)
	for n in range(N):
		gene = newList[n]
		YA, YD, OA, OD = database[gene]
		if OA - YA > 16:
			YA += 24 
			YD += 24
		if YA - OA > 16:
			OA += 24 
			OD += 24

		YF = YD-YA
		OF = OD-OA
		plt.subplot(3,2,3)
		plt.plot([YA,YD],[1.0*n/N,1.0*n/N],'g',linewidth=50/N)
		plt.subplot(3,2,4)
		plt.plot([OA,OD],[1.0*n/N,1.0*n/N],'r',linewidth=50/N)

	plt.subplot(3,1,3)
	plt.plot(master_y+master_y,'g')

	plt.plot(master_o+master_o,'r')

goinfo = []
for go in goDB.keys():
	up, down, no = 0, 0, 0
	for gene in goDB[go]:
		YA, YD, OA, OD = database[gene]
		YF = YD-YA
		OF = OD-OA
		if YF-OF > 2:
			down+=1
		elif YF-OF < -2:
			up += 1
		else:
			no += 1
	goinfo.append([(up),go])
	print (go, ':', up, down, no)

goinfo.sort(reverse=True)
print (goinfo)

#plt.subplot(1,2,1)
plt.xlim(0,36)
plt.ylim(-0.5,1)
#plt.subplot(1,2,2)
plt.xlim(0,36)
plt.ylim(-0.5,1)
newgo = goinfo[4][1]
#14
print (newgo)
#exit()response to light stimulus

BG(goDB['response to light stimulus'])
plt.show()


