import numpy as np
import math

def drawPolarPlot(data, subplot, angleS, color=None, fmt=None, error=None, errorbar=None, fill=None):
	N = len(data)
	xS, yS = [], []
	for n in list(range(N))+[0]:
		x = math.cos(angleS[n])*data[n]
		y = math.sin(angleS[n])*data[n]
		xS.append(x)
		yS.append(y)
		if errorbar:
			dx = math.cos(angleS[n])*error[n]
			dy = math.sin(angleS[n])*error[n]
			ddx = math.cos(angleS[n]+math.pi/2)*0.03*max(data)
			ddy = math.sin(angleS[n]+math.pi/2)*0.03*max(data)
			subplot.plot([x-dx,x+dx],[y-dy,y+dy],'-',color=color)
			subplot.plot([x-dx-ddx,x-dx+ddx],[y-dy-ddy,y-dy+ddy],'-',color=color)
			subplot.plot([x+dx-ddx,x+dx+ddx],[y+dy-ddy,y+dy+ddy],'-',color=color)
		subplot.plot(xS,yS,color+fmt)


def drawPoly(p,subplot,fmt=None,fill=None):
	x,y = p.exterior.xy
	if fill:subplot.fill(x,y,fmt)
	else:subplot.plot(x,y,fmt)


def normalizePolarPlot(subplot):
	a, b = subplot.get_xlim()
	c, d = subplot.get_ylim()
	scale = max(abs(a),abs(b),abs(c),abs(d))
	subplot.axhline(0,linestyle='--', color='black')
	subplot.axvline(0,linestyle='--', color='black')
	subplot.set_xlim((-scale,scale))
	subplot.set_ylim((-scale,scale))

def genSubPlotS(fig):
	plt.clf()
	rawPlot = fig.add_subplot(3,3,1)
	polarPlot = fig.add_subplot(3,3,2)
	polarAreaPlot = fig.add_subplot(3,3,3)
	plotS = [polarPlot,polarAreaPlot,rawPlot,distPlot,scalePlot,scalePolarPlot,anglePolarPlot,angleDistPlot,PhasePlot]
	return plotS

def makeGraph(data, rawPlot,angleS, color):
	N = len(data)
	if not angleS:angleS = [math.pi*2/N*k for k in range(N)]
	meanS= [np.mean(dat) for dat in data]
	stdS = [np.std(dat) for dat in data]
	timeS = np.array(theta2ZT(angleS))
	xTic = list(timeS-48)+list(timeS-24)+list(timeS)+list(timeS+24)
	#print (xTic)
	rawPlot.errorbar(xTic, meanS+meanS+meanS+meanS,yerr=stdS+stdS+stdS+stdS, color=color, capsize=2 ,fmt='o-')
	rawPlot.set_xlim(-10,34)
	#rawPlot.set_ylim(0,)

def theta2ZT(thetaS):
	temp = np.array(thetaS) % (math.pi*2)
	h0 = temp[0]*12/math.pi
	if h0 < 12:result = [h0]
	else:result = [h0-24]
	for t in temp[1:]:
		h = t*12/math.pi
		if h > result[-1]:result.append(h)
		else: result.append(h+24)
	return result
