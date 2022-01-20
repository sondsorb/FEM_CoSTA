import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from utils import length



SBMFACT = [
            (
                lambda x,t,alpha,time_delta: 0,
                lambda x,t,alpha,time_delta: alpha*(t+x*x/2)
            ),
            (
                lambda x,t,alpha,time_delta: 1-alpha,
                lambda x,t,alpha,time_delta: t+alpha*x*x/2
            ),
            (
                lambda x,t,alpha,time_delta: 0.5/(t+alpha+1+time_delta)**0.5-120*x*x-60*x+40,
                lambda x,t,alpha,time_delta: (t+alpha+1+time_delta)**0.5 + 10*x*x*(x-1)*(x+2)
            ),
            (
                lambda x,t,alpha,time_delta: alpha/(t+0.1+time_delta)**2*(x*(1-x)+2*((x-1)*np.tanh(x/(t+0.1+time_delta))-t-0.1+time_delta))*np.cosh(x/(t+0.1+time_delta))**-2,
                lambda x,t,alpha,time_delta: 2+alpha*(x-1)*np.tanh(x/(t+0.1+time_delta))
            ),
            (
                lambda x,t,alpha,time_delta: 2*np.pi*(np.cos(2*np.pi*t+alpha)+2*np.pi*np.sin(2*np.pi*t+alpha))*np.cos(2*np.pi*x) ,
                lambda x,t,alpha,time_delta: 1+np.sin(2*np.pi*t+alpha)*np.cos(2*np.pi*x)
            ),
    ]

# For tuning:
SBMFACT_TUNING = [
            (
                lambda x,t,alpha,time_delta: 0.5/(t+alpha+1+time_delta)**0.5-84*x*x-42*x+28,
                lambda x,t,alpha,time_delta: (t+alpha+1+time_delta)**0.5 + 7*x*x*(x-1)*(x+2)
            ),
            (
                lambda x,t,alpha,time_delta: x**3*(x-alpha)/(t+0.1+time_delta)**2 + (12*x**2-6*alpha*x)/(t+0.1+time_delta),
                lambda x,t,alpha,time_delta: -x**3*(x-alpha)/(t+0.1+time_delta)
            ),
    ]

class Solution:
    def __init__(self, T, f_raw, u_raw, zero_source=True, name='?', time_delta=0):
        self.T = T # final time (default in u)
        self.f_raw = f_raw # raw versions of function are function of (x,t,T,alpha,time_delta)
        self.u_raw = u_raw
        self.zero_source = zero_source
        self.name=name # for plot legends
        self.time_delta = time_delta # time_delta - shift in time, required for using negative times on coarse time grids.

    def set_alpha(self, alpha):
        self.alpha = alpha

    def u(self, x, t=None):
        try:
            if t==None:
                t = self.T
        except:
            pass
        return self.u_raw(x=x, t=t, alpha=self.alpha, time_delta=self.time_delta)

    def f(self, x, t=None):
        if t==None:
            t = self.T
        return 0 if self.zero_source else self.f_raw(x=x, t=t, alpha=self.alpha, time_delta=self.time_delta)


def manufacture_solution(u, t_var, x_vars, alpha_var=None, d1=2, d2=1):
    '''Manufactures f from the 2d heat equation given u (sympy equation)
    returns functions with y=0'''
    assert d1>=d2 and d1<4 and d2>0
    f = u.diff(t_var)
    for d in range(d1):
        f -= u.diff(x_vars[d], x_vars[d])
    f_temp = lambdify([*[x_vars],t_var,alpha_var],f, "numpy")
    u_temp = lambdify([*[x_vars],t_var,alpha_var],u, "numpy")
    def f(x,t,alpha,time_delta=0):
        return f_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def u(x,t,alpha,time_delta=0):
        if length(x[0])>0: # if x is a list, and not a single point, we unpack recursively
            return np.array([u(x_i,t,alpha,time_delta) for x_i in x])
        assert length(x) == d2
        return u_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    return (f,u)
