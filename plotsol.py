import functions
import numpy as np
from matplotlib import pyplot as plt, cm

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
