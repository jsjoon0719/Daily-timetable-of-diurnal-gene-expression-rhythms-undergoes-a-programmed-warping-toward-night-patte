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


def click(event):
	global start_x, start_y
	global end_x, end_y
	global rect
	global canvas
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

def activateCanvas(imgArray):
	global start_x, start_y
	global end_x, end_y
	global rect
	global canvas

	(y,x,_) = np.shape(imgArray)
	img = Image.fromarray(imgArray.astype('uint8'))
	img = ImageTk.PhotoImage(img)
	root = Tk()
	root.title("Cropper")
	root.geometry(str(y)+'x'+str(x))
	canvas = Canvas(root, width = x, height = y)
	canvas.pack()
	circle = 0
	rect = 0
	start_x = None
	start_y = None
	end_x, end_y = None, None
	canvas.create_image(0,0,image=img, anchor = 'nw')
	root.bind('<Button 1>', click)
	root.mainloop()