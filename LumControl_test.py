from tkinter import *
import datetime


timeLineS = [[0,8]]
###Default is Dark

timeRectS = []

def posInCanvas(num):
	return 4+366*num/24

def drawTimeTable(self=None):
	global timeLineS
	global timeRectS
	global canvas
	for timeRect in timeRectS:
		canvas.delete(timeRect)
	timeRectS = []
	timeLineS.sort()
	prev = 0
	n = 0
	for [p,n] in timeLineS:
		if prev < p:
			timeRectS.append(canvas.create_rectangle(4,posInCanvas(prev),150,posInCanvas(p),outline='red',fill='white'))
		timeRectS.append(canvas.create_rectangle(4,posInCanvas(p),150,posInCanvas(n),outline='red',fill='black'))
		prev = n
	if n < 24:
			timeRectS.append(canvas.create_rectangle(4,posInCanvas(n),150,posInCanvas(24),outline='red',fill='white'))

def set_LD(self=None):
	global timeLineS
	timeLineS = [[0,8]]
	drawTimeTable()

def set_SD(self=None):
	global timeLineS
	timeLineS = [[0,12]]
	drawTimeTable()

def set_LL(self=None):
	global timeLineS
	timeLineS = []
	drawTimeTable()

def set_DD(self=None):
	global timeLineS
	timeLineS = [[0,24]]
	drawTimeTable()

def addDark(self=None):
	global timeLineS
	dat = entry_addDark.get()
	try:
		[p,n] = dat.split('-')
		timeLineS.append([float(p),float(n)])
	except:
		return 0
	drawTimeTable()
def excute(self=None):
	global timeLineS
	global rgbON, rgbGO
	global interval
	global exposure_time
	exposure_time = float(entry_exposure.get())
	interval = float(entry_interval.get())
	rgbON = rgbGO.get()
	print (exposure_time, interval, rgbON)
	root.destroy()




root = Tk()
root.geometry('600x500')

rgbGO = IntVar()

panel_LC = Label(root, text='Light Condition(LD as default)')
panel_LC.pack()
panel_LC.place(x=80,y=10)


timeText = ''
for r in range(25):
	timeText += str(r)+':00\n'

panel_time = Label(root, text=timeText)
panel_time.pack()
panel_time.place(x=10,y=40)
canvas = Canvas(root,relief='solid',bd=1)
canvas.pack()
canvas.place(x=50,y=40)
canvas.configure(width=150,height=370)

button_LD = Button(root, text='LD', command=set_LD)
button_LD.pack()
button_LD.place(x=230,y=50,width=100)


button_SD = Button(root, text='SD', command=set_SD)
button_SD.pack()
button_SD.place(x=230,y=80,width=100)



button_LL = Button(root, text='LL', command=set_LL)
button_LL.pack()
button_LL.place(x=230,y=110,width=100)


button_DD = Button(root, text='DD', command=set_DD)
button_DD.pack()
button_DD.place(x=230,y=140,width=100)


panel_addTime = Label(root, text='AddDark\nEx)10.5-16.5\nfor 10:30-16:30')
panel_addTime.pack()
panel_addTime.place(x=230,y=200)

entry_addDark = Entry(root)
entry_addDark.pack()
entry_addDark.place(x=230,y=260,width=100)



button_addDark = Button(root, text='AddDark', command=addDark)
button_addDark.pack()
button_addDark.place(x=230,y=280,width=100)

drawTimeTable()


panel_exposure = Label(root, text='Exposure(sec):')
panel_exposure.pack()
panel_exposure.place(x=350,y=30)

entry_exposure = Entry(root, textvariable='300')
entry_exposure.pack()
entry_exposure.place(x=450,y=30,width=50)

panel_interval = Label(root, text='Interval(sec):')
panel_interval.pack()
panel_interval.place(x=350,y=60)

entry_interval = Entry(root)
entry_interval.pack()
entry_interval.place(x=450,y=60,width=50)

panel_rgb = Label(root, text='RGB scanning after \nluminescence for \nsenescecne assay')
panel_rgb.pack()
panel_rgb.place(x=350,y=90)

import time
time_string = time.strftime('%H:%M:%S')

check_rgb = Checkbutton(root,variable=rgbGO)
check_rgb.pack()
check_rgb.place(x=470,y=90,width=50)

myvar = Label(root, text = 'Current Time(for check PC)\n'+time_string)
myvar.pack()
myvar.place(x=350,y=150)



button_excute = Button(root, text='Execute', command=excute)
button_excute.pack()
button_excute.place(x=380,y=250,width=100,height=100)


root.mainloop()

import time
import win32api
import win32con
import datetime
import os
import time

import tkinter, win32api, win32con, pywintypes

label = tkinter.Label(text = 'Automatic control running:\nPlease do not disrupt anything!', font=('Times New Roman','40'), fg='red', bg='black')
label.master.overrideredirect(True)
label.master.geometry("+750+450")
label.master.lift()
label.master.wm_attributes("-topmost", True)
label.master.wm_attributes("-disabled", True)
label.master.wm_attributes("-transparentcolor", "black")

hWindow = pywintypes.HANDLE(int(label.master.frame(), 16))
# http://msdn.microsoft.com/en-us/library/windows/desktop/ff700543(v=vs.85).aspx
# The WS_EX_TRANSPARENT flag makes events (like mouse clicks) fall through the window.
exStyle = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT
win32api.SetWindowLong(hWindow, win32con.GWL_EXSTYLE, exStyle)

label.pack()
label.update()

def textupdate(givetext, front='Automatic control running:\nPlease do not disrupt anything!\n'):
	label.config(text = front+givetext)
	label.master.overrideredirect(True)
	label.master.geometry("+750+450")
	label.master.lift()
	label.master.wm_attributes("-topmost", True)
	label.master.wm_attributes("-disabled", True)
	label.master.wm_attributes("-transparentcolor", "black")
	label.update()

def currentTime():
	time_string = time.strftime('%H:%M:%S')
	[h,m,s] = time_string.split(':')
	return int(h)+1.0*int(m)/60+1.0*int(s)/3600

if 1:
	nowtime = str(datetime.datetime.now())
	print (nowtime)
	expCode = ''
	for stat in nowtime:
		if stat == ' ':
			expCode += '_'
			continue
		if stat == ':':
			expCode += '-'
		else:
			expCode += stat
	def RGB(step_n):
		print ('rgb')

	def LUM(step_n, LUMtime):
		print ('lum')

	def Light_ON_OFF():
		global currentLight
		currentLight = 1 - currentLight
		print (currentLight, "light")

	def measurement():
		global rgbON
		global step_n
		if rgbON:
			RGB(step_n)
			LUM(step_n,exposure_time)
			step_n+=1
		else:
			LUM(step_n,exposure_time)
			step_n+=1



def mainfun():
	global currentLight
	global step_n
	step_n = 0
	currentLight = 0
	measurement()
	ET = currentTime()
	while 1:
		CT = currentTime()
		if CT-ET > 1.0*interval/60/60:
			measurement()
			ET = CT
		darkNow = 0
		for [n,p] in timeLineS:
			if n <= CT and CT <= p:
				darkNow = 1
		if darkNow == currentLight:
			Light_ON_OFF()

mainfun()