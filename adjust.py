import tkinter as tk
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load
from PIL import ImageTk, Image
import cv2


def go_adjust(event=None):
	global min_entry,mid_entry,max_entry,mainFig,err_entry
	min_value = float(min_entry.get())
	#mid_value = float(mid_entry.get())
	max_value = float(max_entry.get())
	err_value = float(err_entry.get())

	mid = (min_value + max_value)/2

	(Y,X) = np.shape(mat)
	temp = np.zeros((Y,X))
	print(max_value,min_value)
	temp[0,0] = max_value
	temp[0,1] = min_value
	for y in range(Y):
		for x in range(X):
			value = mat[y,x]
			if value:
				temp[y,x] = value
				#if value > mid_value:
				#	temp[y,x] = (value-mid_value)/(max_value-mid_value)*(max_value-mid)+mid
				#if value < mid_value:
				#	temp[y,x] = (-value+mid_value)/(-min_value+mid_value)*(-min_value+mid)+min_value

				if value > max_value:
					temp[y,x] = max_value
				if value < min_value:
					temp[y,x] = min_value
				if value < err_value:
					temp[y,x] = max_value
			else:
				temp[y,x] = None
	plt.clf()
	plt.imshow(temp, cmap=cmap, interpolation='nearest')
	plt.colorbar()
	plt.savefig('temp.png')
	test = cv2.imread('temp.png')
	test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
	test = Image.fromarray(test)
	test = ImageTk.PhotoImage(image=test)
	mainFig.configure(image=test)
	mainFig.image=test




def adjustColor(mat2):
	global min_entry,mid_entry,max_entry, mat, cmap,mainFig,err_entry
	mat = mat2
	root = tk.Tk()
	root.geometry('1200x1000')
	cmap = plt.cm.jet
	cmap.set_bad(color='black')
	plt.imshow(mat, cmap=cmap, interpolation='nearest')
	plt.colorbar()
	plt.savefig('temp.png')
	test = cv2.imread('temp.png')
	test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
	test = Image.fromarray(test)
	test = ImageTk.PhotoImage(image=test)
	mainFig = tk.Label(root,image=test)
	mainFig.pack()
	mainFig.place(x=200,y=10)
	min_entry = tk.Entry(root)
	mid_entry = tk.Entry(root)
	max_entry = tk.Entry(root)
	err_entry = tk.Entry(root)
	go_adjust_button = tk.Button(root,command=go_adjust,text='Adjust')

	min_entry.place(x=50, y= 5, width = 50)
	mid_entry.place(x=50, y =25, width = 50)
	max_entry.place(x=50, y =45, width = 50)
	err_entry.place(x=50, y =65, width = 50)
	go_adjust_button.place(x=50,y=90,width=100)


	tk.mainloop()



def transform(value,min_value,max_value,err_value,midPointS):
	if value < err_value:return None
	if value < min_value:return min_value
	if value > max_value:return max_value
	prevY = min_value
	prevX = min_value
	k = 1
	for midX in midPointS:
		midY = 1.0*(max_value-min_value)*k/(len(midPointS)+1)+min_value
		k += 1
		if value < midX:
			return (midY-prevY)/(midX-prevX)*(value-prevX)+prevY
		#break
		prevY = midY
		prevX = midX
	midY = max_value
	midX = max_value
	return 1.0*(midY-prevY)/(midX-prevX)*(value-prevX)+prevY


def generateMAP(mat,min_value,max_value,err_value,midPointS):
	(Y,X) = np.shape(mat)
	temp = np.zeros((Y,X))
	for y in range(Y):
		for x in range(X):
			value = mat[y,x]
			if value:
				temp[y,x] = 1.0*transform(value,min_value,max_value,err_value,midPointS)
			else:
				temp[y,x] = None

	#temp[0,0] = None
	#temp[0,1] = None	
	temp[0,0] = min_value*1.0
	temp[0,1] = max_value*1.0
	#print(min_value,max_value,temp[0,0],temp[0,1])
	cmap = plt.cm.jet
	cmap.set_bad(color='black')
	plt.imshow(temp, cmap=cmap, interpolation='nearest')
	plt.xticks([])
	plt.yticks([])
	#plt.show()
	plt.savefig('temp.png')
	plt.cla()
	return cv2.imread('temp.png')

import random
from skimage.metrics import structural_similarity
def mse(imageA, imageB):
	err = structural_similarity(imageA,imageB,multichannel=True)
	return err
if __name__ == '__main__':
	[dataStorm, lifegram, periodgram, timeList, p2l, l2p] = load('temp2/1019/11')
	adjustColor(periodgram)
	img1 = generateMAP(lifegram,85,105,30,[])
	#exit()
	min_bound = 0.45
	max_bound = 0.6

	scoreSheet = [[-9999999999999]]

	best = 99999999999
	for nn in range(10000):
		if nn % 500 == 0:print (nn)
		seedS = [random.random() for _ in range(4)]
		seedS.sort()
		min_value = seedS[0]*0.15+min_bound
		max_value = seedS[3]*0.15+min_bound
		midPointS = [seedS[t]*0.15+min_bound for t in range(1,3)]
		img2 = generateMAP(periodgram,min_value,max_value,0,midPointS)
		score =  structural_similarity(img1,img2,multichannel=True)
		#score = np.sum(abs(img1-img2))
		if score > max(scoreSheet)[0]:
			print(score, nn, min_value,max_value,midPointS)
			scoreSheet.append([score,min_value,max_value,midPointS])
			cv2.imwrite('best.png',img2)

	[score,min_value,max_value,midPointS] = min(scoreSheet)
	print(score, nn, min_value,max_value,midPointS)
	img2 = generateMAP(lifegram,min_value,max_value,0,midPointS)
