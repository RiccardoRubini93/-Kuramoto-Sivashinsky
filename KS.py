import numpy as np
import pylab as pl
import pdb

class KS(object):
	#
	# Solution of 1-d Kuramoto-Sivashinsky equation
	# u_t + u*u_x + u_xx + diffusion*u_xxxx = 0, periodic BCs on [0,2*pi*L].
	# time step dt with N fourier collocation points.
	
	def __init__(self,L=16,N=128,dt=0.5,diffusion=1.0):
		
		self.L = L 
		self.n = N 
		self.dt = dt
		self.diffusion = diffusion
		
		kk = N*np.fft.fftfreq(N)[0:int((N/2)+1)]  # wave numbers
		
		self.wavenums = N*np.fft.fftfreq(N)[0:int((N/2)+1)]
		k  = kk.astype(np.float)/L
		
		self.ik    = 1j*k                   # spectral derivative operator
		self.lin   = k**2 - diffusion*k**4  # Fourier multipliers for linear term

		xx = np.linspace(0,L,N)		
		self.xx = xx
        
		#x = 0.1*np.random.rand(N)
		x = np.cos(4*np.pi*xx/L)*(1.0+np.sin(2*np.pi*xx/L))
		#x = np.sin(4*np.pi*xx/L)
        # remove mean from initial condition.
        
		#pdb.set_trace()

		self.x = x - x.mean()
        	
        #spectral space variable
		
		#pdb.set_trace()
		self.xspec = np.fft.rfft(self.x,axis=-1)
    
	def nlterm(self,xspec):
		
		x = np.fft.irfft(xspec,axis=-1)
		
		return -0.5*self.ik*np.fft.rfft(x**2,axis=-1)
    
	def step(self):
		
		# semi-implicit third-order runge kutta update.
		
		self.xspec = np.fft.rfft(self.x,axis=-1)
		
		xspec_save = self.xspec.copy()
        
		for n in range(3):
			
			dt = self.dt/(3-n)
			# explicit RK3 step for nonlinear term
			
			self.xspec = xspec_save + dt*self.nlterm(self.xspec)
			# implicit trapezoidal adjustment for linear term
			
			self.xspec = (self.xspec+0.5*self.lin*dt*xspec_save)/(1.-0.5*self.lin*dt)

		self.x = np.fft.irfft(self.xspec,axis=-1)

	def plot_spectrum(self,u):

		sp = np.sum(abs(np.fft.fft(u)),0)
		k = self.wavenums
	
		pl.figure(1)
		pl.semilogy(k[:-1],sp[0:len(k)-1]/max(sp),"r--",lw = 3)
		pl.xlabel("k",fontsize=12)
		pl.ylim([1e-4,1.2])
		pl.show()
		

		

		

		











