import tkinter as tk
import os
import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import support_1 as sub

targetFolder = (os.listdir('Target'))[0]
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
for n in range(endN):
	fileNtoTime[n] = (fileNtoTime[n]-iniTime)/60/60

endTime = fileNtoTime[endN-1]


root = tk.Tk()
root.title('ColorConvert')
root.geometry('1200x800')


def select(self):
	print ('Target/'+targetFolder+'/' + str(scale.get())+'_730_.tif')
	leftCanvas.itemconfig(leftImage, image = imageDB[scale.get()])
	print ('stop')
	pass





var=tk.IntVar()
scale=tk.Scale(root, variable=var, command=select, orient="horizontal", showvalue=True, tickinterval=50, to=endN-1, length=700)
scale.pack()
scale.place(x=250,y=700)
leftCanvas = tk.Canvas(root, width=550, height=550)
leftCanvas.pack()
leftCanvas.place(x=25,y=25)

imageDB = []
for n in range(endN):
	print (n)
	img = Image.open('Target/'+targetFolder+'/' + str(n) + '_730_.tif')
	testImage = np.array(img)
	testImage2 = sub.resizeArray(testImage/255, 550, 550)
	testImage3 = Image.fromarray(np.array(testImage2))
	imageDB.append(ImageTk.PhotoImage(testImage3))
leftImage = leftCanvas.create_image(0, 0, anchor= tk.NW, image = imageDB[0])

root.mainloop()