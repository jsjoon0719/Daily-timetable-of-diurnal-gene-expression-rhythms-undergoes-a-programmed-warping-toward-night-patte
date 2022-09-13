if 1:
	import numpy as np
	from joblib import dump, load
	import tkinter as tk
	from sklearn.svm import SVC
	from PIL import ImageTk, Image
	from matplotlib import pyplot as plt
	import sys
	from scipy.signal import savgol_filter


N_leaf = 4
resol = 15
cut = 3

folder = 'RawData_0802'
dataStorm = load('temp/'+folder+'/cropped')
(size_Y,size_X) = dataStorm[0][1].shape
print (size_Y, size_X)



global circle
circle = 0

def motion(event):
	 global circle
	 global canvas
	 x, y = event.x, event.y
	 radius = 1
	 canvas.delete(circle)  #to refresh the circle each motion
	 x_max = x + radius
	 x_min = x - radius
	 y_max = y + radius
	 y_min = y - radius
	 circle = canvas.create_rectangle(x_max, y_max, x_min, y_min, outline="white", dash = (3,5), width = 5)

leafPointS = []
blankPointS = []

def leftClick(event):
	global x_range
	global y_range
	global leafPointS
	global frontImg
	radius = 1
	x = event.x
	y = event.y
	x_max = x + radius
	x_min = x - radius
	y_max = y + radius
	y_min = y - radius
	x_range = [x-radius,x+radius]
	y_range = [y-radius,y+radius]
	for dx in range(x_min,x_max):
		for dy in range(y_min,y_max):
			leafPointS.append(list(dataStorm[0][2][dy][dx]))
	canvas.create_rectangle(x_max, y_max, x_min, y_min, outline="blue", dash = (3,5), width = 5)
	if len(leafPointS) and len(blankPointS): learn()


def rightClick(event):
	global x_range
	global y_range
	global leafPointS
	global frontImg
	radius = 1

	x = event.x
	y = event.y
	x_max = x + radius
	x_min = x - radius
	y_max = y + radius
	y_min = y - radius
	
	x_range = [x-radius,x+radius]
	y_range = [y-radius,y+radius]

	for dx in range(x_min,x_max):
		for dy in range(y_min,y_max):
			blankPointS.append(list(dataStorm[0][2][dy][dx]))

	canvas.create_rectangle(x_max, y_max, x_min, y_min, outline="red", dash = (3,5), width = 5)
	if len(leafPointS) and len(blankPointS): learn()



leafArea = []  
clusterN = 0



def chunkFilter(mat, resolution):
	global cut
	global align
	global clusterN

	thre = cut
	yN = len(mat)
	xN = len(mat[0])
	tempMat = np.zeros((yN,xN))
	align = np.zeros((yN,xN))
	templist = dict()
	memberShip = []
	for x in range(xN):
		for y in range(yN):
			if not mat[y][x]:continue
			if [y,x] in align:continue
			memberS = tracker(mat,y,x,0)
			memberShip.append([len(memberS),memberS])
  
	memberShip.sort(reverse=True)

	for mn in range(N_leaf):
		memberS = memberShip[mn][1]
		clusterN += 1
		templist[clusterN] = [[],[]]
		for member in memberS:
					[dy,dx] = member
					tempMat[dy][dx] = clusterN
					templist[clusterN][0].append(dx*resolution)
					templist[clusterN][1].append(dy*resolution)





	clusterInfo = dict()
	tempClusterS = []
	for clusterN in templist.keys():
		xList = templist[clusterN][0]
		yList = templist[clusterN][1]
		xleft = min(xList)
		xright = max(xList)
		yleft = min(yList)
		yright = max(yList)
		x_center = (xleft+xright)/2
		y_center = (yleft+yright)/2
		clusterInfo[clusterN] = [x_center,y_center,xleft,xright,yleft,yright]
		tempClusterS.append([x_center+y_center,clusterN])
	firstCluster = max(tempClusterS)[1]

	clusterNow = firstCluster
	already = [firstCluster]
	cluster2num = dict()
	cluster2num[firstCluster] = 1
	nowfinding = 2
	while 1:
		[xn, yn, _, _, _, _] = clusterInfo[clusterNow]
		#print 'loop 2', clusterNow, xn, yn
		while 1:
			[xn, yn, _, _, _, _] = clusterInfo[clusterNow]
			#print 'loop 1', clusterNow, xn, yn
			temp = []
			for clusterN in templist.keys():
				if clusterN in already:continue
				[xn, yn, _, _, _, _] = clusterInfo[clusterNow]
				[xt, yt, _, _, _, _] = clusterInfo[clusterN]
				if yn < yt:
					continue
				if abs(yt-yn) < abs(xt-xn)*2:
					continue
				temp.append([abs(yt-yn),clusterN])
			if len(temp) == 0:break
			clusterNow = min(temp)[1]
			already.append(clusterNow)
			cluster2num[clusterNow] = nowfinding
			nowfinding += 1


		temp = []
		clusterNow = firstCluster
		for clusterN in templist.keys():
			if clusterN in already:continue
			[xn, yn, _, _, _, _] = clusterInfo[clusterNow]
			[xt, yt, _, _, _, _] = clusterInfo[clusterN]
			temp.append([(yt-yn)**2+(xt-xn)**2,clusterN])
		if len(temp) == 0:break
		clusterNow = min(temp)[1]
		cluster2num[clusterNow] = nowfinding
		already.append(clusterNow)
		nowfinding += 1

	for clusterN in clusterInfo.keys():
		clusterInfo[clusterN].append(cluster2num[clusterN])

	return tempMat, clusterInfo

sys.setrecursionlimit(10000)


def tracker(mat, y, x, recur):
	size_Y = len(mat)
	size_X = len(mat[0])
	global align
	if y >= size_Y or y < 0 or x >=size_X or x < 0:return []
	if align[y,x]:return []
	memberS2 = [[y,x]]
	align[y,x] = 1
	up = []
	down = []
	right = []
	left = []
	if y > 0:
		if mat[y-1,x] and not align[y-1,x]:
			up = tracker(mat,y-1,x,recur+1)
	if y < size_Y-1:
		if mat[y+1,x] and not align[y+1,x]:
			down = tracker(mat,y+1,x,recur+1)
	if x < size_X-1:
		if mat[y,x+1] and not align[y,x+1]:
			right = tracker(mat,y,x+1,recur+1)
	if x > 0:
		if mat[y,x-1] and not align[y,x-1]:
			left = tracker(mat,y,x-1,recur+1)
	return memberS2 + up + down + right + left



def learn(data_n = 0):
	#print 'enter night'
	global leafPointS
	global blankPointS
	global canvas
	global leafArea
	global model
	global final_resultMat, final_clusterInfo
	X = leafPointS + blankPointS
	Y = [1 for _ in leafPointS] + [0 for _ in blankPointS]
	model = SVC(kernel='linear',cache_size=20000).fit(X, Y)
	for leaf in leafArea:
		canvas.delete(leaf)
	leafArea = []
	miniY = int(size_Y/resol)
	miniX = int(size_X/resol)
	resultMat = np.zeros((miniY,miniX))
	for dx in range(miniX):
		for dy in range(miniY):
			result = model.predict([dataStorm[data_n][2][dy*resol][dx*resol]])[0]
			resultMat[dy][dx] = result
	global clusterN
	clusterN = 0
	resultMat, clusterInfo = chunkFilter(resultMat,resol)
	NoOver = []
	for dx in range(miniX):
		for dy in range(miniY):
			if resultMat[dy][dx]:
				cN = resultMat[dy][dx]
				x_min = dx*resol-1
				x_max = dx*resol+1
				y_min = dy*resol-1
				y_max = dy*resol+1
				leafArea.append(canvas.create_rectangle(x_max, y_max, x_min, y_min, outline="blue", width = 1))
	for clusterN in clusterInfo.keys():
		[x_center,y_center,xleft,xright,yleft,yright,cN] = clusterInfo[clusterN]
		leafArea.append(canvas.create_text(x_center, y_center, font='Times 20 bold', text = str(cN)))
	final_resultMat, final_clusterInfo = resultMat, clusterInfo
	return resultMat, clusterInfo

def stage2(event):
	global resol
	resol -= 1
	if resol == 0:
		resol = 1
		print ('limit')
	learn()

def stage3(event):
	global resol
	resol += 1
	learn()

outlineS = []

def learn2(targetImg, targetDat,xleft,yleft):
	global outlineS
	global leafPointS
	global blankPointS
	global canvas
	global leafArea
	global model
	global final_resultMat, final_clusterInfo

	size_Y, size_X = np.shape(targetImg)
	miniY = int(size_Y)
	miniX = int(size_X)

	resultMat = np.zeros((miniY,miniX))

	for dx in range(miniX):
		for dy in range(miniY):
			result = model.predict([targetDat[dy][dx]])[0]
			resultMat[dy][dx] = result

	resultMat = chunkFilter_best(resultMat, targetDat, targetImg, xleft,yleft)
	return resultMat


outlineS = []

maskMatrix = np.zeros((size_Y,size_X))

k1 = 1
import time



def tracker2(mat, y, x, recur, marker):
	global k1
	global align
	global recursiveN
	global unDO
	k1 += 1
	if k1 % 100 == 0:
		print(k1, x, y, 2)
	size_Y = len(mat)
	size_X = len(mat[0])
	align[y,x] = marker
	up = []
	down = []
	right = []
	left = []

	if k1 < 2000:
		if y > 0:
			if mat[y-1,x] and not align[y-1,x]:
				tracker2(mat,y-1,x,recur+1,marker)
		if y < size_Y-1:
			if mat[y+1,x] and not align[y+1,x]:
				tracker2(mat,y+1,x,recur+1,marker)
		if x < size_X-1:
			if mat[y,x+1] and not align[y,x+1]:
				tracker2(mat,y,x+1,recur+1,marker)
		if x > 0:
			if mat[y,x-1] and not align[y,x-1]:
				tracker2(mat,y,x-1,recur+1,marker)
	else:
		if y > 0:
			if mat[y-1,x] and not align[y-1,x]:
				unDO.append([y-1,x])
		if y < size_Y-1:
			if mat[y+1,x] and not align[y+1,x]:
				unDO.append([y+1,x])
		if x < size_X-1:
			if mat[y,x+1] and not align[y,x+1]:
				unDO.append([y,x+1])
		if x > 0:
			if mat[y,x-1] and not align[y,x-1]:
				unDO.append([y,x-1])
	return 0

import random






def chunkFilter_best(mat, targetDat, backImg,xleft,yleft):
	global align
	global clusterN
	global outlineS
	global maskMatrix
	global cN
	global recursiveN
	global unDO

	size_Y, size_X = np.shape(mat)
	mini_Y, mini_X = np.shape(backImg)
	print('best', xleft, xleft+size_X, yleft,yleft+size_Y)
	yN = len(mat)
	xN = len(mat[0])
	tempMat = np.zeros((yN,xN))
	align = np.zeros((yN,xN))
	templist = dict()

	#print ('ssssss244455551111')
	tempList = []
	leafMarker = 1
	for x in range(xN):
		#print (x, xN)
		for y in range(yN):
			if not mat[y][x]:continue
			if align[y,x]:continue
			#print (x,y)
			leafMarker += 1
			while 1:
				k1 = 1
				unDO = []
				tracker2(mat,y,x,0,leafMarker)
				if len(unDO) > 0:
					DO = []
					for [dy, dx] in unDO:
						if not align[dy,dx]:
							tracker2(mat,dy,dx,0,leafMarker)
				else:break
			memberS = []
			for dx in range(xN):
				for dy in range(yN):
					if align[dy,dx] == leafMarker:
						memberS.append([dy,dx])

			tempList.append([len(memberS),memberS])
			#print (tempList)
	result = max(tempList)[1]

	#for [y,x] in [max(result), min(result)]:
	#	blankPointS.append(list(targetDat[y,x]))

	print ('ssssss24445555')

	for [y,x] in result:
		tempMat[y,x] = 1
	plt.subplot(3,2,1)
	plt.xlim(0,size_X)
	plt.ylim(0,size_Y)
	plt.imshow(tempMat)

	xDict = dict()
	for [y,x] in result:
		if x in xDict.keys():
			xDict[x].append(y)
		else:
			xDict[x] = [y]
	xList = list(xDict.keys())
	xList.sort()
	upper = []
	lower = []
	for x in xList:
		ytemp = xDict[x]
		upper.append(max(ytemp))
		lower.append(min(ytemp))

	plt.subplot(3,2,3)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.plot(xList,upper,'r')
	plt.plot(xList,lower,'b')

	print ('ssssss244455555566665')
	yDict = dict()
	for [y,x] in result:
		if y in yDict.keys():
			yDict[y].append(x)
		else:
			yDict[y] = [x]
	yList = list(yDict.keys())
	yList.sort()
	lefter = []
	righter = []
	for y in yList:
		xtemp = yDict[y]
		lefter.append(max(xtemp))
		righter.append(min(xtemp))


	plt.plot(lefter,yList,'c')
	plt.plot(righter,yList,'y')


	plt.xlim(0,size_X)
	plt.ylim(0,size_Y)


	plt.subplot(3,2,5)
	plt.gca().set_aspect('equal', adjustable='box')

	plt.xlim(0,size_X)
	plt.ylim(0,size_Y)
	import random
	#plt.savefig(str(random.random())+'.png')

	upper2 = savgol_filter(upper, 17, 2)
	lower2 = savgol_filter(lower, 17, 2)
	lefter2 = savgol_filter(lefter, 17, 2)
	righter2 = savgol_filter(righter, 17, 2)
	plt.plot(xList,upper2,'r--')
	plt.plot(xList,lower2,'b--')
	plt.plot(lefter2,yList,'c--')
	plt.plot(righter2,yList,'y--')

	print ('ssssss2444')

	plt.subplot(3,2,2)
	plt.gca().set_aspect('equal', adjustable='box')

	xN = len(xList)
	point1 = xList[int(xN/4)]
	point2 = xList[int(xN/4*3)]
	newxList = []
	newupper = []
	newlower = []
	newlefter = []
	newrighter = []
	newylist = []

	cordinateS = []

	for n in range(len(xList)):
		x = xList[n]
		if x >= point1 and x <= point2:
			newxList.append(x)
			newlower.append(round(lower2[n]))
			newupper.append(round(upper2[n]))
			cordinateS.append([int(round(lower2[n],0)),x])
			cordinateS.append([int(round(upper2[n],0)),x])
	for n in range(len(yList)):
		y = yList[n]
		x1 = lefter[n]
		x2 = righter[n]
		if x2 <= point1 or x1 >= point2:
			newylist.append(y)
			newlefter.append(round(lefter2[n]))
			newrighter.append(round(righter2[n]))
			cordinateS.append([y,int(round(lefter2[n],0))])
			cordinateS.append([y,int(round(righter2[n],0))])

	print ('ssssss2222')

	plt.plot(newxList,newupper,'r--')
	plt.plot(newxList,newlower,'b--')
	plt.plot(newlefter,newylist,'c--')
	plt.plot(newrighter,newylist,'y--')

	plt.xlim(0,size_X)
	plt.ylim(0,size_Y)
	global newImg
	overlayed = backImg[:]*1.5

	plt.subplot(3,2,4)
	for [y,x] in cordinateS:
		overlayed[y,x] = 255
	plt.imshow(overlayed)

	import random
	cordiDict = dict()
	for [y,x] in cordinateS:
		if not y in  cordiDict.keys():
			cordiDict[y] = []
		cordiDict[y].append(x)
		outlineS.append(canvas.create_oval(x+xleft, y+yleft, x+xleft, y+yleft, width = 0, fill = 'green'))


	newMat = np.zeros((size_Y,size_X))
	mask = np.zeros((mini_Y,mini_X))
	mask2 = np.zeros((mini_Y,mini_X,4))
	for y in range(size_Y):
		for x in range(size_X):
			newMat[y,x] = None
			if not y in cordiDict.keys():continue
			if x >= min(cordiDict[y]) and x <= max(cordiDict[y]):
				newMat[y,x] = backImg[y,x]
				maskMatrix[y+yleft,x+xleft] = 1
				mask[y,x] = 1
				mask2[y,x,:] = [1,1,1,1]

	ministorm = []
	for n in range(len(dataStorm)):
		temp1 = dataStorm[n][0] 
		temp2 = dataStorm[n][1][yleft:yleft+mini_Y,xleft:xleft+mini_X] * mask
		temp3 = dataStorm[n][2][yleft:yleft+mini_Y,xleft:xleft+mini_X] * mask2
		ministorm.append([temp1,temp2,temp3])


	print ('ssssss1111')
	dump(ministorm, 'temp/'+folder+'/Leaf'+str(cN)+'.bin')

	plt.subplot(3,2,6)
	cmap = plt.cm.OrRd
	cmap.set_bad(color='black')
	plt.imshow(newMat,cmap=cmap)


	#plt.savefig('temp/'+folder+'/'+str(random.random())+'.png')
	plt.clf()
	cN += 1
	return tempMat

def finish(event):
	global final_resultMat, final_clusterInfo
	global outlineS
	global maskMatrix
	global cN
	cN = 0
	maskMatrix = np.zeros((size_Y,size_X))
	for out in outlineS:
		canvas.delete(out)
	outlineS = []
	print (final_clusterInfo)
	clusterS = list(final_clusterInfo.keys())
	clusterS.sort()
	for c_num in clusterS:
		print (c_num)
		clusterInfo = final_clusterInfo[ c_num]
		[x_center,y_center,xleft,xright,yleft,yright,cN] = clusterInfo
		yleft = max([0,yleft-2*resol])
		yright = min([size_Y-1,yright+int(1.4*resol)])
		xleft = max([0,xleft-2*resol])
		xright = min([size_X-1,xright+int(1.4*resol)])
		newImg = frontMat[yleft:yright,xleft:xright]
		newDat = dataStorm[0][2][yleft:yright,xleft:xright]
		newMask = learn2(newImg,newDat,xleft,yleft)
		plt.subplot(2,len(clusterS),2*(c_num-1)+1)
		plt.imshow(newMask)
		plt.subplot(2,len(clusterS),2*(c_num-1)+2)
		plt.imshow(newImg)
		plt.clf()
	plt.clf()
	plt.imshow(maskMatrix)
	plt.show()
	print (final_clusterInfo)



if 1:
	frontMat = dataStorm[0][1]
	frontMat = frontMat/np.mean(frontMat)*100
	root = tk.Tk()
	root.title("LeafFinder")
	root.geometry(str(size_X)+'x'+str(size_Y))
	canvas = tk.Canvas(root, width = size_X, height = size_Y)
	canvas.pack()
	frontImg = ImageTk.PhotoImage(Image.fromarray(frontMat))
	canvas.create_image(0,0,image=frontImg, anchor = 'nw')
	canvas.bind("<Button 1>",leftClick)
	root.bind("<Right>",stage2)

	root.bind("<Left>",stage3)
	canvas.bind("<Button 3>",rightClick)
	root.bind('<Motion>', motion)
	root.bind('<q>', finish)
	root.bind('<Q>', finish)
	root.mainloop()





