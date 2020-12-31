import numpy as np
from KS import KS
import pylab as pl
import matplotlib.animation as animation
import sys

L   = int(sys.argv[1])/(2*np.pi)  # domain is 0 to 2.*np.pi*L
N   = int(sys.argv[1])      # number of collocation points
dt  = 0.5          # time step
diffusion = 1.0

ks = KS(L=L,diffusion=diffusion,N=N,dt=dt) # instantiate model

x = np.arange(N)*(L/N)

n = 1000
nmin=500
uu = [] 
tt = []
vspec = np.zeros(ks.xspec.shape[0], np.float)
#x = np.arange(N)

fig, ax = pl.subplots(1)
line, = ax.plot(x, ks.x.squeeze(),lw=5)
ax.set_xlim(0,L)
ax.set_ylim(-5,5)

#Init only required for blitting to give a clean slate.

def init():
    global line
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

# cut off the transient
for n in range(nmin):
    ks.step()

def updatefig(n):
    
	global tt,uu,vspec
	ks.step()
	vspec += np.abs(ks.xspec.squeeze())**2
	u = ks.x.squeeze()
	line.set_ydata(u)
	pl.title('Time = ' + str(n*dt))	
	pl.xlabel('X')
	pl.ylabel('u')	

	print(n)
	
	uu.append(u) 
	tt.append(n*dt)
	return line,

ani = animation.FuncAnimation(fig, updatefig, np.arange(1,n+1), init_func=init,interval=25, blit=True,repeat=False)
#ani.save('KS_animation.mp4', fps=40, extra_args=['-vcodec', 'libx264'])
#ani.save('KS_animation.mp4', fps=40)

pl.show()

pl.figure(2)

# make contour plot of solution, plot spectrum.

ncount = len(uu)
vspec = vspec/ncount
uu = np.array(uu) 
tt = np.array(tt)

pl.contourf(x,tt[:n],uu[:n],1001,cmap=pl.cm.magma)
pl.xlabel('x')
pl.ylabel('t')
pl.colorbar()
pl.title('Solution of the K-S equation')


pl.show()

#save results

#np.savetxt('U_KS.txt', np.asmatrix(uu), delimiter=' ')
