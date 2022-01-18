import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify



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

def manufacture_solution(u, t_var, x_var, y_var, alpha_var):
    '''Manufactures f from the 2d heat equation given u (sympy equation)
    returns functions with y=0'''
    f = u.diff(t_var) - u.diff(x_var,x_var) - u.diff(y_var,y_var)
    time_delta = sp.symbols('time_delta')
    return (
            lambdify([x_var,t_var,alpha_var,time_delta],f.subs(y_var,0), "numpy"),
            lambdify([x_var,t_var,alpha_var,time_delta],u.subs(y_var,0), "numpy")
           )
