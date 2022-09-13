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
import finder
import time

import Tkinter, win32api, win32con, pywintypes

label = Tkinter.Label(text = 'Automatic control running:\nPlease do not disrupt anything!', font=('Times New Roman','40'), fg='red', bg='black')
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
	textupdate('Calibrating...','')
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
	os.mkdir('C:\\AutoScrapper\\'+expCode)
	#exit()

	VK_CODE = {'backspace':0x08,
			           'tab':0x09,
			           'clear':0x0C,
			           'enter':0x0D,
			           'shift':0x10,
			           'ctrl':0x11,
			           'alt':0x12,
			           'pause':0x13,
			           'caps_lock':0x14,
			           'esc':0x1B,
			           'spacebar':0x20,
			           'page_up':0x21,
			           'page_down':0x22,
			           'end':0x23,
			           'home':0x24,
			           'left_arrow':0x25,
			           'up_arrow':0x26,
			           'right_arrow':0x27,
			           'down_arrow':0x28,
			           'select':0x29,
			           'print':0x2A,
			           'execute':0x2B,
			           'print_screen':0x2C,
			           'ins':0x2D,
			           'del':0x2E,
			           'help':0x2F,
			           '0':0x30,
			           '1':0x31,
			           '2':0x32,
			           '3':0x33,
			           '4':0x34,
			           '5':0x35,
			           '6':0x36,
			           '7':0x37,
			           '8':0x38,
			           '9':0x39,
			           'a':0x41,
			           'b':0x42,
			           'c':0x43,
			           'd':0x44,
			           'e':0x45,
			           'f':0x46,
			           'g':0x47,
			           'h':0x48,
			           'i':0x49,
			           'j':0x4A,
			           'k':0x4B,
			           'l':0x4C,
			           'm':0x4D,
			           'n':0x4E,
			           'o':0x4F,
			           'p':0x50,
			           'q':0x51,
			           'r':0x52,
			           's':0x53,
			           't':0x54,
			           'u':0x55,
			           'v':0x56,
			           'w':0x57,
			           'x':0x58,
			           'y':0x59,
			           'z':0x5A,
			           'numpad_0':0x60,
			           'numpad_1':0x61,
			           'numpad_2':0x62,
			           'numpad_3':0x63,
			           'numpad_4':0x64,
			           'numpad_5':0x65,
			           'numpad_6':0x66,
			           'numpad_7':0x67,
			           'numpad_8':0x68,
			           'numpad_9':0x69,
			           'multiply_key':0x6A,
			           'add_key':0x6B,
			           'separator_key':0x6C,
			           'subtract_key':0x6D,
			           'decimal_key':0x6E,
			           'divide_key':0x6F,
			           'F1':0x70,
			           'F2':0x71,
			           'F3':0x72,
			           'F4':0x73,
			           'F5':0x74,
			           'F6':0x75,
			           'F7':0x76,
			           'F8':0x77,
			           'F9':0x78,
			           'F10':0x79,
			           'F11':0x7A,
			           'F12':0x7B,
			           'F13':0x7C,
			           'F14':0x7D,
			           'F15':0x7E,
			           'F16':0x7F,
			           'F17':0x80,
			           'F18':0x81,
			           'F19':0x82,
			           'F20':0x83,
			           'F21':0x84,
			           'F22':0x85,
			           'F23':0x86,
			           'F24':0x87,
			           'num_lock':0x90,
			           'scroll_lock':0x91,
			           'left_shift':0xA0,
			           'right_shift ':0xA1,
			           'left_control':0xA2,
			           'right_control':0xA3,
			           'left_menu':0xA4,
			           'right_menu':0xA5,
			           'browser_back':0xA6,
			           'browser_forward':0xA7,
			           'browser_refresh':0xA8,
			           'browser_stop':0xA9,
			           'browser_search':0xAA,
			           'browser_favorites':0xAB,
			           'browser_start_and_home':0xAC,
			           'volume_mute':0xAD,
			           'volume_Down':0xAE,
			           'volume_up':0xAF,
			           'next_track':0xB0,
			           'previous_track':0xB1,
			           'stop_media':0xB2,
			           'play/pause_media':0xB3,
			           'start_mail':0xB4,
			           'select_media':0xB5,
			           'start_application_1':0xB6,
			           'start_application_2':0xB7,
			           'attn_key':0xF6,
			           'crsel_key':0xF7,
			           'exsel_key':0xF8,
			           'play_key':0xFA,
			           'zoom_key':0xFB,
			           'clear_key':0xFE,
			           '+':0xBB,
			           ',':0xBC,
			           '-':0xBD,
			           '.':0xBE,
			           '/':0xBF,
			           '`':0xC0,
			           ';':0xBA,
			           '[':0xDB,
			           '\\':0xDC,
			           ']':0xDD,
			           "'":0xDE,
			           '`':0xC0}

	def press(*args):
		    '''
		    one press, one release.
		    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
		    '''
		    for i in args:
		        win32api.keybd_event(VK_CODE[i], 0,0,0)
		        time.sleep(.02)
		        win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)
	def press_string(dat):
			time.sleep(0.2)
			for da in dat:
				if da == '$':
					win32api.keybd_event(VK_CODE['ctrl'], 0 ,0)
					press('a')
					win32api.keybd_event(VK_CODE['ctrl'], 0 ,win32con.KEYEVENTF_KEYUP ,0)
					continue
				if da == ':':
					win32api.keybd_event(VK_CODE['shift'], 0 ,0)
					press(';')
					win32api.keybd_event(VK_CODE['shift'], 0 ,win32con.KEYEVENTF_KEYUP ,0)
					continue
				if da == '_':
					win32api.keybd_event(VK_CODE['shift'], 0 ,0)
					press('-')
					win32api.keybd_event(VK_CODE['shift'], 0 ,win32con.KEYEVENTF_KEYUP ,0)
					continue
				if da == '(':
					win32api.keybd_event(VK_CODE['shift'], 0 ,0)
					press('9')
					win32api.keybd_event(VK_CODE['shift'], 0 ,win32con.KEYEVENTF_KEYUP ,0)
					continue
				if da == ')':
					win32api.keybd_event(VK_CODE['shift'], 0 ,0)
					press('0')
					win32api.keybd_event(VK_CODE['shift'], 0 ,win32con.KEYEVENTF_KEYUP ,0)
					continue


				if da.isupper():
					win32api.keybd_event(VK_CODE['shift'], 0 ,0)
					press(da.lower())
					win32api.keybd_event(VK_CODE['shift'], 0 ,win32con.KEYEVENTF_KEYUP ,0)
				else:
					press(da)



	def ClickButton(button):
		time.sleep(0.1)
		if testdrive:buttonlib[button] = finder.tracker(button)
		win32api.SetCursorPos(buttonlib[button])
		time.sleep(0.1)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
		#win32api.SetCursorPos((1058, 766))
		time.sleep(0.1)

	def aquireImage(exposure_time, filename):
		ClickButton('exposure')
		time.sleep(0.1)
		press_string('$')
		time.sleep(0.1)
		#ClickButton('exposure')
		press_string(str(exposure_time))
		ClickButton('acquire')
		time.sleep(exposure_time/1000+2)
		ClickButton('binning')
		ClickButton('close_win')
		#exit()
		time.sleep(0.3)
		ClickButton('savewin_yes')
		time.sleep(0.5)
		press_string('C:\\AutoScrapper\\'+expCode+'\\'+ filename)
		ClickButton('save_click')


	def RGB(step_n):
		global LOG
		textupdate('RGB scanning')
		ClickButton('open')
		time.sleep(8)
		ClickButton('450')
		aquireImage(1000,str(step_n)+'_450_'+str(k))
		ClickButton('450')
		ClickButton('660')
		aquireImage(1000,str(step_n)+'_660_'+str(k))
		ClickButton('660')
		ClickButton('730')
		aquireImage(1000,str(step_n)+'_730_'+str(k))
		ClickButton('730')
		ClickButton('close')
		nowtime = str(datetime.datetime.now())
		LOG.write('RGB measurement at ' + nowtime+'\n')
		time.sleep(8)

	def LUM(step_n, LUMtime):
		global LOG
		textupdate('Measuring luminscence...')
		ClickButton('open')
		time.sleep(8)
		aquireImage(LUMtime*1000,str(step_n)+'_lum')
		ClickButton('close')
		nowtime = str(datetime.datetime.now())
		LOG.write('LUM measurement at ' + nowtime+'\n')
		time.sleep(8)

	def Light_ON_OFF():
		global currentLight, LOG
		time.sleep(1)
		ClickButton('white')
		time.sleep(1)
		currentLight = 1 - currentLight
		nowtime = str(datetime.datetime.now())
		LOG.write('Light state change to ' +str(currentLight) + ' at ' + nowtime + '\n')

	def measurement():
		global rgbON
		global step_n
		global LOG
		if rgbON:
			RGB(step_n)
			LUM(step_n,exposure_time)
			step_n+=1
		else:
			LUM(step_n,exposure_time)
			step_n+=1



def mainfun():
	global step_n
	global currentLight
	step_n = 0
	currentLight = 0
	measurement()
	ET = currentTime()
	LOG = open('Log_'+expCode+'.txt','w')
	LOG.write('Excuted at ' +expCode+'\n')
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

buttonlib = dict()

from calibrate import calibration 


testdrive = 1

getcom = calibration('Control.jpg')
for get in getcom:
	[command,x,y] = get
	buttonlib[command] = [x,y]


getcom = calibration('Acquire.jpg')
for get in getcom:
	[command,x,y] = get
	buttonlib[command] = [x,y]
ClickButton('exposure')
time.sleep(0.1)
press_string('$')
time.sleep(0.1)
press_string('1000')
ClickButton('acquire')
time.sleep(3)


getcom = calibration('Closewin2.jpg')
for get in getcom:
	[command,x,y] = get
	buttonlib[command] = [x,y]
ClickButton('close_win')

getcom = calibration('Yes.jpg')
for get in getcom:
	[command,x,y] = get
	buttonlib[command] = [x,y]
time.sleep(0.3)
ClickButton('savewin_yes')
time.sleep(0.5)

getcom = calibration('Text.jpg')
for get in getcom:
	[command,x,y] = get
	buttonlib[command] = [x,y]

filename = 'test'
press_string('C:\\AutoScrapper\\'+expCode+'\\'+ filename)
ClickButton('save_click')

testdrive = 0
mainfun('PROGRAM.TXT')