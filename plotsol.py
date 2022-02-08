import functions
import FEM
import numpy as np
from matplotlib import pyplot as plt, cm

def sbmfact():
    #for sol in [1,2,3,4]:
    for sol in [0,1]:
        alpha = 1
        T = 5
        
        #f, u = functions.SBMFACT[sol]
        f, u = functions.SBMFACT_TUNING[sol]
        s = functions.Solution(T, f, u, False)
        s.set_alpha(alpha)
        
        N=1000
        X = np.array([np.linspace(0,1,N+1)]*(N+1)).T
        T = np.array([np.linspace(0,5,N+1)]*(N+1))
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X,T,s.u(X,T), cmap=cm.coolwarm)
        #ax.plot_surface(X,T,u(X,T))
        ax.view_init(elev=24, azim=-30)
        ax.set_xlabel('x')
        ax.set_ylabel('time')
        ax.set_zlabel('u')
        plt.tight_layout()
        #plt.savefig(f'../preproject/1d_heat_figures/sols/{sol}.pdf')
        plt.show()

def elsols():
    for sol in [0,1,2]:
        alpha = 1
        T = 1
        N = 20

        u,t,x,a = functions.ELsols[sol].values()
        f,u,w = functions.manufacture_elasticity_solution(u, x, t, alpha_var=a, d1=2, d2=2, nu=0.25)
        #u,t,x,a = functions.ELsols3d[sol]
        #f,u,w = functions.manufacture_elasticity_solution(u, x, t, alpha_var=a, d1=3, d2=2, nu=0.25)

        s = functions.Solution(T,f,u,w_raw = w)
        s.set_alpha(alpha)
        X = np.array([np.linspace(-1,1,N+1)]*(N+1)).T
        Y = np.array([np.linspace(-1,1,N+1)]*(N+1))
        #disc = FEM.Disc(T, time_steps=1, Ne=N, equation='elasticity', p=1, dim=2, xa=-1, xb=1, ya=-1, yb=1)
        fig = plt.figure()
        #ax1 = fig.add_subplot(121)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        pts = np.array([X,Y]).T
        im1 = ax1.imshow(s.u(pts,t=T)[:,:,0], cmap=cm.coolwarm)
        im2 = ax2.imshow(s.u(pts,t=T)[:,:,1], cmap=cm.coolwarm)
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.show()


if __name__ == '__main__':
    #sbmfact()
    elsols()
