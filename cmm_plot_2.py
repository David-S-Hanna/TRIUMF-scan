# routine to read and plot data from 
# the TRIUMF coordinate measuring machine (CMM)
#
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import glob as gb
import pylab as py
import matplotlib
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
font = {'size'   : 8}
matplotlib.rc('font', **font)
#
# set up some arrays
#
mu=np.ones(10)
dmu=np.ones(10)
sg=np.ones(10)
dsg=np.ones(10)
#
# define functions
#
# plot points with errors
#
def plotGraph(x,y,yerr,xerr):
      x
      y
      xerr
      yerr
      plot = plt.errorbar(x, y, yerr, xerr, capsize=0, fmt='.')
      return fig
#
def plotpoints(nx,ny,nn,data,xl1,xl2,xl3,xl4,yl1,yl2,yl3,yl4,nb,title):
      x = data[0]
      y = data[1]
      z = data[2]
      fig=plt.figure()
      plt.subplot(nx,ny,nn)
      plt.title(title)
      dx = 0.1*(x/x)
      dy = dx 
#
# plot the x and y values
#
      plt.xlim([xl1,xl2])
      plt.ylim([yl1,yl2])
      plot1 = plotGraph(x, y, dy, dx)
      plt.xlabel('x-position (mm)',fontdict=font)
      plt.ylabel('y-position (mm)')
#
# histogram the z values
#
      plt.subplot(nx,ny,nn+1)
      plt.xlim([xl3,xl4])
      plt.ylim([yl3,yl4])
      plt.hist(z, bins=nb, range=(xl3,xl4))
      plt.xlabel('z-value (mm)',fontdict=font)
      plt.ylabel('entries')      
      pp.savefig()
      return
#
#
def xrow(data, irow, yl1, yl2, zl1, zl2, title):
#
# function to plot the z value of 
# scan points in a row at constant x
# 
      datax=data[1]
      datay=data[2]
      ir1 = irow*5
      ir2 = ir1+5
      xpl = datax[ir1:ir2]
      ypl = datay[ir1:ir2]
      dxpl = 0.001*(xpl/xpl)
      dypl = dxpl
      plt.xlim([yl1,yl2])
      plt.ylim([zl1,zl2])
      plot1 = plotGraph(xpl, ypl, dypl ,dxpl)
      return
#
# main program
#
#
# start with a clean slate
#
plt.close("all")
#
# get the data
#
# read the metal data into a one-dimensional array
#
metal = np.genfromtxt('aerogel-metal.dat', unpack=True)
#
# change the array into a two-dimensional array
#
metal.shape = (3,120)
#
# read the aerogel top data into a one-dimensional array
#
top = np.genfromtxt('aerogel-top.dat', unpack=True)
#
# change the array into a two-dimensional array
#
top.shape = (3,26)
#
# read the aerogel bottom data into a one-dimensional array
#
bottom = np.genfromtxt('aerogel-bottom.dat', unpack=True)
#
# change the array into a two-dimensional array
#
bottom.shape = (3,23)
#
# plot the x and y coordinates of the measurements
#
pp = PdfPages("cmm_2.pdf")
#
#
# metal points
#
plot1 = plotpoints(2,2,1,metal,0.,160.,19.0,19.15,0.,160.,0.0,10.0,100,"Metal")
#
# aerogel top points
#
plot1 = plotpoints(2,2,1,top,0.,160.,13.0,14.0,0.,160.,0.0,10.0,100,"Aerogel Top")
#
# aerogel bottom points
#
plot1 = plotpoints(2,2,1,bottom,0.,160.,3.0,4.0,0.,160.,0.0,10.0,100,"Aerogel Bottom")
# 
# loop over top and bottom and over lines of constant x 
# and plot the z values
#
# top
#
fig=plt.figure()
plt.subplot(3,2,1)
plt.title("Aerogel Top")
plot1 = xrow(top,0,25.,125.,13.,14.,'row-0') 
plt.tick_params(labelbottom='off')
plt.ylabel('z-position (mm)')
plt.subplot(3,2,2)
plot1 = xrow(top,1,25.,125.,13.,14.,'row-0') 
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.subplot(3,2,3)
plot1 = xrow(top,2,25.,125.,13.,14.,'row-0') 
plt.tick_params(labelbottom='off')
plt.ylabel('z-position (mm)')
plt.subplot(3,2,4)
plot1 = xrow(top,3,25.,125.,13.,14.,'row-0') 
plt.tick_params(labelleft='off')
plt.xlabel('y-position (mm)',fontdict=font)
plt.subplot(3,2,5)
plot1 = xrow(top,4,25.,125.,13.,14.,'row-0') 
plt.xlabel('y-position (mm)',fontdict=font)
plt.ylabel('z-position (mm)')
pp.savefig()
#
# bottom
#
fig=plt.figure()
plt.subplot(3,2,1)
plt.title("Aerogel Bottom")
plot1 = xrow(bottom,0,25.,125.,3.,4.,'row-0') 
plt.tick_params(labelbottom='off')
plt.ylabel('z-position (mm)')
plt.subplot(3,2,2)
plot1 = xrow(bottom,1,25.,125.,3.,4.,'row-0') 
plt.tick_params(labelleft='off')
plt.xlabel('y-position (mm)',fontdict=font)
plt.subplot(3,2,3)
plot1 = xrow(bottom,2,25.,125.,3.,4.,'row-0') 
plt.xlabel('y-position (mm)',fontdict=font)
plt.ylabel('z-position (mm)')
pp.savefig()
#
# now plot the differences in x, y, and z for the 
# first 15 measurements, top and bottom
#
xpl=np.arange(15)+1.0
dxpl=0.001*(xpl/xpl)
dypl=dxpl
fig=plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.title("Aerogel Top-Bottom")
m1=top[0][0:15]
m2=bottom[0][0:15]
ypl=m1-m2
plot1 = plotGraph(xpl, ypl, dypl ,dxpl)
plt.xlim(0.,16.)
plt.ylim(-2.,2.)
plt.tick_params(labelbottom='off')
plt.ylabel('x difference (mm)')
plt.subplot(3,1,2)
plt.grid(True)
m1=top[1][0:15]
m2=bottom[1][0:15]
ypl=m1-m2
plot1 = plotGraph(xpl, ypl, dypl ,dxpl)
plt.xlim(0.,16.)
plt.ylim(-2.,2.)
plt.tick_params(labelbottom='off')
plt.ylabel('y difference (mm)')
plt.subplot(3,1,3)
plt.grid(True)
m1=top[2][0:15]
m2=bottom[2][0:15]
ypl=m1-m2
plot1 = plotGraph(xpl, ypl, dypl ,dxpl)
plt.xlim(0.,16.)
plt.ylim(9.5,10.5)
plt.ylabel('z difference (mm)')
plt.xlabel('data point number')
pp.savefig()
pp.close()
print "done"
