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
      return
#
#
def residuals(data, yl1, yl2, title):
      irow = 0
      jrow = 0
      ipl = 0
      x = np.arange(11)+0.5
      zr = np.ones(11)
      z = data[2]
      w = data[3]
      res = np.ones(len(z))
      fig=plt.figure()
      while irow<11:
            i1 = irow*11
            i2 = i1+11
            zr = z[i1:i2]
            wt = w[i1:i2]
            a = np.polyfit(x, zr, 5, w=wt)
            zf = np.polyval(a, x)
            rs = zr - zf
            xpl = np.linspace(0, 11, 100)
            ypl = np.polyval(a, xpl)
            plt.subplot(3, 4, irow+1)
            plt.plot(xpl, ypl, 'r--', x, zr, 'bo', markersize=4)
            plt.xlim([0.,11.])
            plt.ylim([yl1, yl2])
            plt.xlabel(irow,fontdict=font)
            if irow < 8:
              plt.tick_params(labelbottom='off')
            if jrow not in [0,4,8]:
              plt.tick_params(labelleft='off')
            res[i1:i2] = rs
            jrow += 1
            irow += 1
      plt.subplot (3,4,1)
      plt.title(title)
      pp.savefig()
      res = res*1000
      fig=plt.figure()
      plt.subplot (2,2,1)
      plt.title(title)
      plt.hist(res, bins=50, range=(-25.0,25.0))
      plt.ylim([0,20])
      plt.xlabel('residuals (um)',fontdict=font)
      plt.ylabel('entries')
      pp.savefig()
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
# read the metal side 1 data into a one-dimensional array
#
data_m1 = np.genfromtxt('metal-side-1.txt', unpack=True)
#
# change the array into a two-dimensional array
#
data_m1.shape = (3,80)
#
# read the aerogel side 1 data into a one-dimensional array
#
data_a1 = np.genfromtxt('aerogel-side-1.txt', unpack=True)
#
# change the array into a two-dimensional array
#
data_a1.shape = (4,121)
#
# read the metal side 2 data into a one-dimensional array
#
data_m2 = np.genfromtxt('metal-side-2.txt', unpack=True)
#
# change the array into a two-dimensional array
#
data_m2.shape = (3,80)
#
# read the aerogel side 2 data into a one-dimensional array
#
data_a2 = np.genfromtxt('aerogel-side-2.txt', unpack=True)
#
# change the array into a two-dimensional array
#
data_a2.shape = (4,121)
#
# plot the x and y coordinates of the measurements
#
pp = PdfPages("cmm.pdf")
#
# start a figure
#
fig=plt.figure()
#
# metal side 1 points
#
plot1 = plotpoints(2,2,1,data_m1,0.,160.,19.0,19.15,0.,160.,0.0,10.0,100,"Side-1")
#
# aerogel side 1 points
#
plot1 = plotpoints(2,2,3,data_a1,0.,160.,13.0,14.0,0.,160.,0.0,10.0,100," ")
pp.savefig()
#
# second page
#
fig=plt.figure()
#
# metal side 2 points
#
plot1 = plotpoints(2,2,1,data_m2,0.,160.,19.0,19.15,0.,160.,0.0,10.0,100,"Side-2")
#
# aerogel side 2 points
#
plot1 = plotpoints(2,2,3,data_a2,0.,160.,16.5,17.5,0.,160.,0.0,10.0,100," ")
pp.savefig()
#
# loop over the rows in the tile scan and fit a polynomial
# then check the residuals from the fit to get an idea of the 
# point-to-point variation
# 
# do side 1 then side 2
#
plot1 = residuals(data_a1, 13.0, 14.0,"Side-1")
plot1 = residuals(data_a2, 16.5, 17.5,"Side-2")
pp.close()
print "done"


