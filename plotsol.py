import functions
import numpy as np
from matplotlib import pyplot as plt

sol = 4
alpha = 1
T = 5

f, u = functions.SBMFACT[sol]
s = functions.Solution(T, f, u, False)
s.set_alpha(alpha)

N=100
X = np.array([np.linspace(0,1,N+1)]*(N+1)).T
T = np.array([np.linspace(0,5,N+1)]*(N+1))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,T,s.u(X,T))
#ax.plot_surface(X,T,u(X,T))
plt.show()
