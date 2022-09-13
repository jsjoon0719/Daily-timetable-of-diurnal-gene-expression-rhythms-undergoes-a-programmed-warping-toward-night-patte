import tkinter as tk
import os
import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import support_1 as sub
from Useful_legacy import *

targetFolder = '0401'
fileS  = os.listdir('Target/'+targetFolder)
headerS = []
buildTimeS = []
fileNtoTime = dict()
for file in fileS:
	header = int(file.split('_')[0])
	if not header in fileNtoTime.keys():
		ctime = os.path.getmtime('Target/'+targetFolder+'/'+file)
		fileNtoTime[header] = ctime

iniTime = fileNtoTime[0]
endN = len(fileNtoTime.keys())
timeS = []
for n in range(endN):
	fileNtoTime[n] = (fileNtoTime[n]-iniTime)/60/60
	timeS.append(fileNtoTime[n])
endTime = fileNtoTime[endN-1]

root = tk.Tk()
root.title('ColorConvert')
root.geometry('600x800')
import time
def select(self=None):
	global newimage
	#print ('Target/'+targetFolder+'/' + str(scale.get())+'_730_.tif')
	newimage = array2image(imageDB[scale.get()])
	#print ('22')
	#time.sleep(0.1)
	leftCanvas.itemconfig(leftImage, image = newimage)
	timeshow.configure(text = str(fileNtoTime[scale.get()]))
	#print ('stop')
	pass

def rightMove(event=None):
	if scale.get() < endN:
		scale.set(scale.get()+1)

def leftMove(event=None):
	if scale.get() > 0:
		scale.set(scale.get()-1)

def array2image(array):
	testImage = Image.fromarray(np.array(array).astype(np.uint8))
	return (ImageTk.PhotoImage(testImage))

def dist(a,b):
	[x1,y1] = a
	[x2,y2] = b
	length = ((x1-x2)**2+(y1-y2)**2)**0.5
	return length

pre = None
def click(event):
	global result, pre
	x,y = event.x, event.y
	if not pre:
		pre = [x,y]
		imageDB[scale.get()][y-2:y+2,x-2:x+2] = [255,0,0]
	else:
		post = [x,y]
		imageDB[scale.get()][y-2:y+2,x-2:x+2] = [0,255,0]
		result[scale.get()].append(dist(pre,post))
		print(result)
		print (dist(pre,post))
		pre = None

	select()

def export():
	with open('result_width.txt', 'w') as F:
		for temp in result:
			for length in temp:
				F.write(str(length)+'\t')
			F.write('X\n')
	print ('Export Done')


var=tk.IntVar()
scale=tk.Scale(root, variable=var, command=select, orient="horizontal", showvalue=True, tickinterval=50, to=endN-1, length=500)
scale.pack()
scale.place(x=50,y=640)


button_rightMove = tk.Button(root, overrelief="solid", width=15, command=rightMove, repeatdelay=1000, repeatinterval=100, text='->')
button_rightMove.pack()
button_rightMove.place(x=400,y=600)


button_leftMove = tk.Button(root, overrelief="solid", width=15, command=leftMove, repeatdelay=1000, repeatinterval=100, text='<-')
button_leftMove.pack()
button_leftMove.place(x=80,y=600)


button_export = tk.Button(root, overrelief="solid", width=15, command=export, repeatdelay=1000, repeatinterval=100, text='Export')
button_export.pack()
button_export.place(x=250,y=600)



timeshow = tk.Label(root,text='0')
timeshow.pack()
timeshow.place(x=250,y=720)

leftCanvas = tk.Canvas(root, width=550, height=550)
leftCanvas.bind("<Button-1>", click)
root.bind("<Right>", rightMove)
root.bind("<Left>", leftMove)
leftCanvas.pack()
leftCanvas.place(x=25,y=25)


imageDB = []
from matplotlib import pyplot as plt
substract = []
imgArrayS = []
pre = np.array([0])
from scipy import stats

totalN = 5
result = [[] for _ in range(totalN)]
for n in range(totalN):
	print (n)
	img1 = Image.open('Target/'+targetFolder+'/' + str(n) + '_450_.tif')
	img2 = Image.open('Target/'+targetFolder+'/' + str(n) + '_660_.tif')
	img3 = Image.open('Target/'+targetFolder+'/' + str(n) + '_730_.tif')
	testImage1 = sub.resizeArray(np.array(img1)/255.,550,550)
	testImage2 = sub.resizeArray(np.array(img2)/255.,550,550)
	testImage3 = sub.resizeArray(np.array(img3)/255.,550,550)
	(x,y) = np.shape(testImage1)
	totalImage = np.zeros((x,y,3))
	totalImage[:,:,0] = testImage1
	totalImage[:,:,1] = testImage2
	totalImage[:,:,2] = testImage3
	#testImage3 = Image.fromarray(np.array(totalImage).astype(np.uint8))
	imageDB.append(totalImage)


from sklearn.decomposition import PCA

newimage = array2image(imageDB[0])
leftImage = leftCanvas.create_image(0, 0, anchor= tk.NW, image = newimage)

root.mainloop()