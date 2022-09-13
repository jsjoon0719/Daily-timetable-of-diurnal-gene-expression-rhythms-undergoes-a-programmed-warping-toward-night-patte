if 1:
	from matplotlib import pyplot as plt
	import os
	from tkinter import *
	import numpy as np
	from PIL import Image, ImageTk, ImageEnhance
	import datetime
	import cv2
	from joblib import dump, load
	from sklearn.svm import SVC
	import sys
	from scipy.signal import savgol_filter
	import math

radius = 10
folder = '1019'
fileList = os.listdir(folder)
masterFolder = os.getcwd()
os.chdir(folder)
fileList.sort(key=os.path.getmtime)
os.chdir(masterFolder)
channelS = []
newfileList = []
fileRecordS = []
for file in fileList:
	if len(file.split('_')) > 2:
		frontID = file.split('_')[0]+'&'+file.split('_')[1]
		if frontID in fileRecordS:continue
		fileRecordS.append(frontID)
	newfileList.append(file)

fileList = newfileList

for file in fileList:
	if 'test' in file:continue 
	channel_type = file.split('_')[1]
	channelS.append(channel_type)
channelS = list(set(channelS))
N_channel = len(channelS)

print(N_channel, channelS)

with Image.open(folder+'/'+fileList[0]) as img:
	Time_start = datetime.datetime.strptime(img.tag[306][0], '%Y%m%d %H:%M:%S.%f')

lumImage = np.array(Image.open(folder+'/'+fileList[0]))
lumImage2 = np.array(Image.open(folder+'/'+fileList[1]))
lumImage3 = np.array(Image.open(folder+'/'+fileList[2]))


(y,x) = np.shape(lumImage)
deterMat = np.zeros((y,x,N_channel))
for k in range(N_channel):
	temp = np.array(Image.open(folder+'/'+fileList[k]))
	deterMat[:,:,k] = temp


circle = 0
rect = 0
start_x = None
start_y = None
end_x, end_y = None, None

def motion(event):
	x, y = event.x, event.y
	if not start_x:return 0
	global circle
	global canvas
	global radius
	global rect
	canvas.delete(circle)
	if not rect:circle = canvas.create_rectangle(start_x, start_y, x, y, outline="blue", dash = (3,5), width = 5)

def click(event):
	global start_x, start_y
	global end_x, end_y
	global rect
	if not rect:
		if start_y:
			rect = canvas.create_rectangle(start_x, start_y, event.x, event.y, outline="red", dash = (3,5), width = 5)
			end_x = event.x
			end_y = event.y

			tempX = [start_x,end_x]
			tempY = [start_y,end_y]
			start_x = min(tempX)
			start_y = min(tempY)
			end_x = max(tempX)
			end_y = max(tempY)
		else:
			start_x = event.x
			start_y = event.y
	else:
		canvas.delete(rect)
		start_x = None
		start_y = None
		end_x = None
		end_y = None
		rect = None


maxSize = 1080
def leftClick(event):
	global leafPointS, blankPointS, scaleFactor, miniMat
	radius = 1
	x = int(event.x/scaleFactor)
	y = int(event.y/scaleFactor)
	x_max = x + radius
	x_min = x - radius
	y_max = y + radius
	y_min = y - radius
	for dx in range(x_min,x_max):
		for dy in range(y_min,y_max):
			leafPointS.append(list(miniMat[dy][dx]))
	canvas2.create_rectangle(x_max*scaleFactor, y_max*scaleFactor, x_min*scaleFactor, y_min*scaleFactor, outline="blue", dash = (3,5), width = 5)
	if len(leafPointS) and len(blankPointS):learn2()

poly = 0
def leftClick_neo(event):
	global leafPointS, blankPointS, scaleFactor, miniMat, poly, prev
	x = int(event.x/scaleFactor)
	y = int(event.y/scaleFactor)
	leafPointS.append([y,x])
	#del(poly)
	if len(leafPointS) > 2:
		temp = []
		for [dy,dx] in leafPointS:
			temp += [dx*scaleFactor,dy*scaleFactor]
		canvas2.delete(poly)
		poly = canvas2.create_polygon(tuple(temp), outline="green", dash = (3,5), width = 5, fill='')
		learn3()

	else:
		0
		#poly = 0

	prev = canvas2.create_rectangle(x*scaleFactor, y*scaleFactor, x*scaleFactor, y*scaleFactor, outline="red", dash = (3,5), width = 5)
	#poly = 0
	if len(leafPointS) and len(blankPointS):learn2()


prev = 0

def rightClick_neo(event):
	global leafPointS, blankPointS, scaleFactor, miniMat, poly, prev
	x = int(event.x/scaleFactor)
	y = int(event.y/scaleFactor)
	leafPointS = leafPointS[0:-1]
	#del(poly)
	if len(leafPointS) > 2:
		temp = []
		for [dy,dx] in leafPointS:
			temp += [dx*scaleFactor,dy*scaleFactor]
		canvas2.delete(poly)
		poly = canvas2.create_polygon(tuple(temp), outline="green", dash = (3,5), width = 5, fill='')
		print ('gogo')
		learn3()
		print ('gogo')
	else:
		0
		#poly = 0

	canvas2.delete(prev)
	#canvas2.create_rectangle(x*scaleFactor, y*scaleFactor, x*scaleFactor, y*scaleFactor, outline="red", dash = (3,5), width = 5)
	#poly = 0
	if len(leafPointS) and len(blankPointS):learn2()


def learn2(data_n = 0):
	global leafPointS, blankPointS, canvas2, miniMat
	global leafArea
	global model
	global final_resultMat, final_clusterInfo
	global sub_x, sub_y
	X = leafPointS + blankPointS
	Y = [1 for _ in leafPointS] + [0 for _ in blankPointS]
	model = SVC(kernel='linear',cache_size=20000).fit(X, Y)
	for leaf in leafArea:
		canvas2.delete(leaf)
	leafArea = []
	(miniY, miniX, _) = np.shape(miniMat)
	resultMat = np.zeros((miniY,miniX))
	for dx in range(miniX):
		for dy in range(miniY):
			result = model.predict([miniMat[dy,dx]])[0]
			resultMat[dy][dx] = result

	global clusterN
	clusterN = 0

	chunkFilter_best(resultMat, miniMat)

subLeafS = []
def learn3(data_n = 0):
	global leafPointS, blankPointS, canvas2, miniMat
	global leafArea
	global model
	global final_resultMat, final_clusterInfo
	global sub_x, sub_y
	global leafPointS
	global resultMat
	global subLeafS

	print('learn3')
	temp = []
	for [y,x] in leafPointS:
		temp.append((y,x))
	polygon = Polygon(temp)

	for leaf in leafArea:
		canvas2.delete(leaf)

	leafArea = []

	(miniY, miniX, _) = np.shape(miniMat)
	resultMat = np.zeros((miniY,miniX))
	print('learn3')

	for leaf in subLeafS:
		canvas2.delete(leaf)
	subLeafS = []
	for dx in range(miniX):
		for dy in range(miniY):
			result = polygon.contains(Point(dy,dx))
			resultMat[dy][dx] = result
			if result:
				subLeafS.append(canvas2.create_rectangle(dx*scaleFactor, dy*scaleFactor, dx*scaleFactor, dy*scaleFactor, outline="red", dash = (3,5), width = 1))




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

sys.setrecursionlimit(10000)

def chunkFilter_best(mat, targetDat):
	global align
	global clusterN
	global outlineS
	global maskMatrix
	global cN
	global recursiveN
	global unDO
	global canvas2
	global cordinateS, cordiDict, size_Y, size_X

	for line in outlineS:
		canvas2.delete(line)
	outlineS = []

	size_Y, size_X = np.shape(mat)
	tempMat = np.zeros((size_Y, size_X))
	align = np.zeros((size_Y, size_X))
	templist = dict()
	tempList = []
	leafMarker = 1
	for x in range(size_X):
		for y in range(size_Y):
			if not mat[y,x]:continue
			if align[y,x]:continue
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
			for dx in range(size_X):
				for dy in range(size_Y):
					if align[dy,dx] == leafMarker:
						memberS.append([dy,dx])

			tempList.append([len(memberS),memberS])

	result = max(tempList)[1]


	for [y,x] in result:
		tempMat[y,x] = 1


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
	import random
	#plt.savefig(str(random.random())+'.png')

	upper2 = savgol_filter(upper, 17, 4)
	lower2 = savgol_filter(lower, 17, 4)
	lefter2 = savgol_filter(lefter, 17, 4)
	righter2 = savgol_filter(righter, 17, 4)

	upper2 = upper
	lower2 = lower
	lefter2 = lefter
	righter2 = righter

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

	cordiDict = dict()
	for [y,x] in cordinateS:
		if not y in  cordiDict.keys():
			cordiDict[y] = []
		cordiDict[y].append(x)
		outlineS.append(canvas2.create_oval(x*scaleFactor, y*scaleFactor, x*scaleFactor, y*scaleFactor, width = 4, fill = 'green', outline="yellow"))

	print( 'Finished')


leafN = 0

leafInfoS = []


def finish_subwin(event):
	global cordinateS
	global cordiDict, leafN
	global start_x, start_y, newWin, size_Y, size_X
	global leafInfoS
	global resultMat
	leafN += 1
	tempxList = []
	tempyList = []
	tempxyList = []
	(size_Y,size_X) = np.shape(resultMat)
	for y in range(size_Y):
		for x in range(size_X):
			if resultMat[y,x]:
				canvas.create_oval(start_x+x, start_y+y, start_x+x, start_y+y, width = 4, fill = 'green', outline="yellow")

	x_min, x_max = start_x, size_X+start_x
	y_min, y_max = start_y, size_Y+start_y

	for y in range(size_Y):
		for x in range(size_X):
			if resultMat[y,x]:
				tempxyList.append([x-x_min +start_x+1, y-y_min+start_y+1])


	canvas.create_rectangle(x_min,y_min,x_max,y_max,outline='yellow')
	(canvas.create_text(start_x,start_y, font='Times 20 bold', text = 'Leaf# '+str(leafN), fill = 'yellow'))
	newWin.destroy()
	leafInfoS.append([leafN,(x_min-1,y_min-1,x_max+2,y_max+2),tempxyList[:]])



def settingFinished(event):
	global start_x, start_y
	global end_x, end_y
	global scaleFactor
	global leafPointS, blankPointS, miniMat, canvas2, leafArea
	global sub_x, sub_y, k1
	global outlineS, newWin
	k1 = 0
	leafPointS = []
	blankPointS = []
	outlineS = []
	leafArea = []
	newWin = Toplevel(root)
	print (start_x, start_y,end_x, end_y)
	size_X = abs(end_x-start_x)
	size_Y = abs(end_y-start_y)
	print (np.shape(lumImage))

	newVisual = np.zeros((size_Y,size_X,3))
	newimg1 = rescaleMat((lumImage)[start_y:end_y, start_x:end_x])
	newimg2 = rescaleMat((lumImage2)[start_y:end_y, start_x:end_x])
	newimg3 = rescaleMat((lumImage3)[start_y:end_y, start_x:end_x])
	newVisual[:,:,0] = newimg1
	newVisual[:,:,1] = newimg2
	newVisual[:,:,2] = newimg3


	#newimg = rescaleMat(newimg,20)
	#saveSample(newVisual)
	newimg = Image.fromarray(newVisual.astype('uint8'))
	print (np.shape(newimg))
	scaleFactor = maxSize/max([size_Y,size_X])
	(sub_x, sub_y) = (int(size_X*scaleFactor),int(size_Y*scaleFactor))
	newWin.geometry(str(sub_x)+'x'+str(sub_y))
	newimg = newimg.resize((sub_x, sub_y))
	newimg = ImageTk.PhotoImage(newimg)
	newimg = newimg
	miniMat = deterMat[start_y:end_y, start_x:end_x]
	canvas2 = Canvas(newWin, width = sub_x, height = sub_y)
	canvas2.pack()
	canvas2.create_image(0,0,image=newimg, anchor = 'nw')
	canvas2.bind("<Button 1>",leftClick_neo)
	canvas2.bind("<Button 3>",rightClick_neo)
	newWin.bind('<q>', finish_subwin)
	newWin.bind('<Q>', finish_subwin)
	newWin.bind('<e>', fatality)
	newWin.mainloop()

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def fatality(event):
	global leafPointS
	temp = []
	for [y,x] in leafPointS:
		temp.append((y,x))
	Polygon(temp)







def fatal_go(cordinateS):
	global cordiDict, leafN
	global start_x, start_y, newWin, size_Y, size_X
	global leafInfoS
	from scipy.interpolate import splprep, splev
	global miniMat
	cy = 0
	cx = 0
	for [y,x] in cordinateS:
		cy += y
		cx += x
	cy = int(cy/len(cordinateS))
	cx = int(cx/len(cordinateS))
	newCordinateS = []

	xS = []
	yS = []
	for [ty,tx] in cordinateS:
		center = np.array([cy,cx])
		target = np.array([ty,tx])
		vector = target-center
		[y,x] = vector
		vector = vector/max([abs(y), abs(x)])
		centerDat = miniMat[cy,cx]
		k = 0
		temp = []
		tempcordi = []
		while 1:
			try:
				k += 1
				[ny,nx] = center+vector*k
				[ny,nx] = [int(ny),int(nx)]
				pointDat = miniMat[ny,nx]
				diff = np.sum((centerDat - pointDat)**2)
				temp.append(diff)
				tempcordi.append([ny,nx])
			except:
				break
		N = len(temp)
		window = 3
		scoreSheet = []
		for k in range(window,N-3*window):
			[ny,nx] = tempcordi[k]
			dist = (ny-ty)**2+(nx-tx)**2
			if dist > (window**2) * 3:continue 
			before = np.sum(temp[k-window:k])
			after = np.sum(temp[k+1:k+window+1])
			scoreSheet.append([abs(before-after),k])

		[ny,nx] = tempcordi[max(scoreSheet)[1]]
		xS.append(nx)
		yS.append(ny)
		newCordinateS.append([ny,nx])
		#(canvas2.create_oval(nx*scaleFactor, ny*scaleFactor, nx*scaleFactor, ny*scaleFactor, width = 4, fill = 'green', outline="green"))
	temp = []
	for [y,x] in newCordinateS:
		dy = y - cy
		dx = x - cx
		cosvalue = dx / (dx**2+dy**2)**0.5
		angle = math.acos(cosvalue)
		if dy < 0:angle = -angle + 2*math.pi
		temp.append([angle,y,x])
	temp.sort()
	#print (temp)
	newCordinateS = []
	for [a, y, x] in temp:
		newCordinateS.append([y,x])
	N = len(temp)
	neoCordinateS = []
	for k in range(N-1):
		[py,px] = newCordinateS[k]
		[ay,ax] = newCordinateS[k+1]
		scale = max([abs(py-ay),abs(px-ax)])
		neoCordinateS.append([py,px])
		if scale <= 1:
			continue
		vector = np.array([ay-py,ax-px])/scale
		origin = np.array([py,px])
		for kk in range(1,scale):
			[ny,nx] = origin + vector*kk
			ny = int(ny)
			nx = int(nx)
			neoCordinateS.append([ny,nx])
	if 0:
		[py,px] = newCordinateS[N-1]
		[ay,ax] = newCordinateS[0]
		scale = max([abs(py-ay),abs(px-ax)])
		neoCordinateS.append([py,px])
		if scale > 1:
			vector = np.array([ay-py,ax-px])/scale
			origin = np.array([py,px])
			for kk in range(1,scale):
				[ny,nx] = origin + vector*kk
				ny = int(ny)
				nx = int(nx)
				neoCordinateS.append([ny,nx])








	newCordinateS = np.array(neoCordinateS)

	return newCordinateS





visualShock = np.zeros((x,y,3))

def rescaleMat(mat, factor = 100):
	flatter = mat.flatten()
	maxele = max(flatter)
	minele = min(flatter)
	midele = np.median(flatter)
	flatter.sort()
	midele = flatter[int(len(flatter)*0.9)]

	return (mat-minele)/(midele-minele)*factor

import random
def save(event):
	global leafInfoS, deterMat
	for [leafN, (start_x,start_y,end_x,end_y), xyList] in leafInfoS:
		xLength = end_x - start_x
		yLength = end_y - start_y
		print(leafN, start_x, start_y, end_x, end_y, xLength, yLength)
		rgbDict = dict()
		lumDict = dict()
		lumTagS = []
		tempMask = np.zeros((yLength,xLength))
		for [x,y] in xyList:
			tempMask[y,x] = 1


		for file in fileList:
			if random.random() > 0.9:print ('Reading files...', file, 'leaf#', leafN)
			tag = file.split('_')[0]+'_'+file.split('_')[1]
			with Image.open(folder+'/'+file) as img:
				delta = (datetime.datetime.strptime(img.tag[306][0], '%Y%m%d %H:%M:%S.%f')) - Time_start
				#print (delta.total_seconds()/60/60, datetime.datetime.strptime(img.tag[306][0], '%Y%m%d %H:%M:%S.%f'), Time_start, file)
				TAE = delta.total_seconds()/60/60
				if TAE < 0:
					print (file)
					continue
				tag = tag + '@' + str(TAE)
				img2 = (np.array(img)[start_y:end_y,start_x:end_x])*tempMask
				if len(file.split('_')) > 2:
						if tag in rgbDict.keys():
							rgbDict[tag].append((img2))
						if not tag in rgbDict.keys():
							rgbDict[tag] = [(img2)]
				else:
					if 'lum' in file or 'LUM' in file:
						lumDict[tag] = (img2)
						lumTagS.append([TAE, tag])
					else:
						if tag in rgbDict.keys():
							rgbDict[tag].append((img2))
						if not tag in rgbDict.keys():
							rgbDict[tag] = [(img2)]
						
		
		rgbimageDict = dict()

		usedLum = []

		for tag in rgbDict.keys():
			if random.random() > 0.9:print('Merging files...', tag, 'leaf#', leafN)
			temp = np.array(rgbDict[tag])
			channel = tag.split('_')[1].split('@')[0]
			[name, TAE] = tag.split('@')
			TAE = float(TAE)
			if channel in rgbimageDict.keys():
				rgbimageDict[channel].append([TAE,np.median(temp, axis =0)])
			else:
				rgbimageDict[channel] = [[TAE,np.median(temp, axis =0)]]


		dataDict = dict()
		new_channelS = list(rgbimageDict.keys())
		new_channelS.sort()
		for channel in new_channelS:
			usedLum = []
			print (channel)
			for [TAE,img] in rgbimageDict[channel]:
				temp = []
				for [lumTAE, lumTag] in lumTagS:
					if not lumTag in usedLum:
						temp.append([(float(lumTAE)-TAE)**2,lumTag])
				if len(temp):
					[_, lumtag] = min(temp)
					if lumtag in usedLum:continue
					usedLum.append(lumtag)
					if lumtag in dataDict.keys():
						dataDict[lumtag].append(img)
					else:
						dataDict[lumtag] = [img]


		dataStrom = []
		n = 0
		for lumtag in dataDict.keys():
			TAE = float(lumtag.split('@')[1])
			lumImage = lumDict[lumtag]
			RGBs = np.array(dataDict[lumtag])
			(channelN, yLength, xLength) = (np.shape(RGBs))
			rgbImage = np.zeros((yLength, xLength, channelN))
			for y in range(yLength):
				for x in range(xLength):
					for c in range(channelN):
						rgbImage[y][x][c] = RGBs[c][y][x]

			dataStrom.append([TAE,lumImage,rgbImage])
			if 0:
				tempImg = Image.fromarray(rgbImage.astype('uint8'))
				Z = rgbImage.reshape((-1,3))
				Z = np.float32(Z)
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
				K = 8
				ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
				center = np.uint8(center)
				res = center[label.flatten()]
				res2 = res.reshape((rgbImage.shape))
				plt.subplot(1,3,1)
				plt.imshow(res2)
				plt.subplot(1,3,2)
				plt.imshow(tempImg)
				plt.subplot(1,3,3)
				plt.imshow(lumImage)
				plt.savefig(str(TAE)+'.png')

			
			print(lumtag, len(dataDict[lumtag]))
		try:os.mkdir('temp/'+folder)
		except:pass
		dump(dataStrom, 'temp/'+folder+'/leaf_'+str(leafN))
		saveNow()
		print ('Complete!')


def saveNow():
	import io
	import PIL.ImageGrab as ImageGrab
	global folder   
	global canvas
	x = root.winfo_rootx() + canvas.winfo_x()
	y = root.winfo_rooty() + canvas.winfo_y()
	xx = x + canvas.winfo_width()
	yy = y + canvas.winfo_height()
	ImageGrab.grab(bbox=(x, y, xx,     yy)).save(folder+"_Named.gif")


def saveSample(data):
	dump(data,'sample.bin')



if 1:
	root = Tk()
	root.title("Cropper")
	root.geometry(str(y)+'x'+str(x))
	canvas = Canvas(root, width = x, height = y)
	canvas.pack()


	lumImage = rescaleMat(lumImage)
	lumImage2 = rescaleMat(lumImage2)
	lumImage3 = rescaleMat(lumImage3)
	for xn in range(x):
		for yn in range(y):
			visualShock[yn,xn,0] = lumImage[yn,xn]
			visualShock[yn,xn,1] = lumImage2[yn,xn]
			visualShock[yn,xn,2] = lumImage3[yn,xn]


	img = Image.fromarray(visualShock.astype('uint8'))
	img = ImageTk.PhotoImage(img)

	canvas.create_image(0,0,image=img, anchor = 'nw')
	root.bind('<Motion>', motion)
	root.bind('<Button 1>', click)
	root.bind('<Return>', settingFinished)
	root.bind('<R>',save)
	root.bind('<r>',save)
	root.mainloop()