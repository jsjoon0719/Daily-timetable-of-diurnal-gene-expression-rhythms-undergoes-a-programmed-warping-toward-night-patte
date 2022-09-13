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
		#if abs(OF-YF) < 1:continue
		#if OF-YF < -4:continue
		temp.append([YA+YD,YA-YD,OD-OA,gene])

	total = master_y + master_o

	tre = max(total)/2
	ydat = 0 
	odat = 0
	tempy = []
	tempo = []
	for n in range(48):
		if master_y[n] > tre:
			ydat+=1
			tempy.append(1)
		else:
			tempy.append(0)

		if master_o[n] > tre:
			odat+=1
			tempo.append(1)
		else:

			tempo.append(0)
	return [ydat]+[odat]+tempy+tempo

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

newgo = goinfo[4][1]
#14
print (newgo)
#exit()response to light stimulus
with open('gg.txt', 'w') as F:
	for go in goDB.keys():
		result = BG(goDB[go])
		F.write(go+'\t')
		for r in result:
			F.write(str(r)+'\t')
		F.write('\n')

