with open('DATA_full.txt') as F:
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
		if YD < YA:YD = YD + 24
		if OD < OA:OD = OD + 24
		geneS.append(gene)
		database[gene] = [YA, YD, OA, OD]

with open('go2.txt') as F:
	F.readline()
	goDB = dict()
	while 1:
		line = F.readline()
		if not line:break
		spline = line.strip('\n').split('\t')
		go = spline[0]
		genes = spline[1]
		if not genes in database.keys():continue
		if go in goDB.keys():
			goDB[go].append(genes)
		else:
			goDB[go] = [genes]

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
		if abs(OF-YF) < 1:continue
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
		#if abs(OF-YF) < 2:continue
		if OF-YF < -1:continue
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
		if 1:
			if (OA+OD) - (YD+YA) > 16:
				OA -= 24 
				OD -= 24
			if - (OA+OD) + (YD+YA) > 16:
				OA += 24 
				OD += 24
		plt.subplot(2,2,1)
		plt.plot([YA,YD],[1.0*n/N,1.0*n/N],'g',linewidth=120/N)
		#plt.plot([YA-24,YD-24],[1.0*n/N,1.0*n/N],'g',linewidth=100/N)
		#plt.plot([YA+24,YD+24],[1.0*n/N,1.0*n/N],'g',linewidth=100/N)
		plt.subplot(2,2,2)
		plt.plot([OA,OD],[1.0*n/N,1.0*n/N],'r',linewidth=120/N)
		#plt.plot([OA-24,OD-24],[1.0*n/N,1.0*n/N],'r',linewidth=100/N)
		#plt.plot([OA+24,OD+24],[1.0*n/N,1.0*n/N],'r',linewidth=100/N)

	plt.subplot(2,1,2)
	for n in range(48):
		print( master_y[n], master_o[n])
	plt.plot([t/2 for t in range(96)], master_y+master_y,'g')

	plt.plot([t/2 for t in range(96)],master_o+master_o,'r')

goinfo = []
for go in goDB.keys():
	up, down, no = 0, 0, 0
	for gene in goDB[go]:
		if not gene in database.keys():continue
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


#exit()response to light stimulus

BG(goDB['Promote'])
plt.subplot(2,2,1)
plt.xlim(0,36)
plt.xticks([0,8,16,24,32])
plt.yticks([])
plt.ylim(-0.1,1.1)
plt.subplot(2,2,2)
plt.xlim(0,36)
plt.xticks([0,8,16,24,32])
plt.ylim(-0.1,1.1)
plt.yticks([])
plt.subplot(2,1,2)
plt.xticks([0,8,16,24,32,40,48])
plt.yticks([])
plt.show()


