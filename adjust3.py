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
	cmap = plt.cm.gist_rainbow
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



if __name__ == '__main__':
	[dataStorm, lifegram, periodgram, timeList, p2l, l2p] = load('temp2/RawData_0624/1')
	adjustColor(periodgram)
