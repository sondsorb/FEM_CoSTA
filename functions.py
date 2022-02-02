import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from utils import length


# Sindre Blakseth's manufactured solutions:
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
    def __init__(self, T, f_raw, u_raw, zero_source=True, name='?', time_delta=0, w_raw=None):
        self.T = T # final time (default in u)
        self.f_raw = f_raw # raw versions of function are function of (x,t,T,alpha,time_delta)
        self.u_raw = u_raw
        self.w_raw = w_raw
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

    def w(self, x, t=None):
        try:
            if t==None:
                t = self.T
        except:
            pass
        return self.w_raw(x=x, t=t, alpha=self.alpha, time_delta=self.time_delta)

def manufacture_solution(u, t_var, x_vars, alpha_var=None, d1=2, d2=1):
    '''Manufactures f from the 2d or 3d heat equation given u (sympy equation)
    returns functions with unused dimensions =0 (eg z=0 for 3d-2d)'''
    print(d1,d2)
    assert d1>=d2
    assert d1<4
    assert d2>0
    f = u.diff(t_var)
    for d in range(d1):
        f -= u.diff(x_vars[d], x_vars[d])
    print('f:', f)
    f_temp = lambdify([*[x_vars],t_var,alpha_var],f, "numpy")
    u_temp = lambdify([*[x_vars],t_var,alpha_var],u, "numpy")
    def f(x,t,alpha,time_delta=0):
        x = [x] if length(x) == 0 else x
        return f_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def u(x,t,alpha,time_delta=0):
        if length(x)>d2: # if x is a list, and not a single point, we unpack recursively
            return np.array([u(x_i,t,alpha,time_delta) for x_i in x])
        assert (length(x)) == (d2 if d2>1 else 0)
        x = [x] if length(x) == 0 else x
        return u_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    return (f,u)

# Elasticity solutions:
t = sp.symbols('t')
x = sp.symbols('x')
y = sp.symbols('y')
alpha = sp.symbols('alpha')
ELsols = [
            {'u':
                    [sp.sin(np.pi*(x+alpha*y))*sp.cos(alpha*t),
                     sp.cos(np.pi*(x+alpha*y))*sp.sin(alpha*t)],
             't_var':t,
             'x_vars':[x,y],
             'alpha_var':alpha,
            },
            {'u':
                    [sp.exp((-t*x**2 + y**2)/(1+alpha+t**2)),
                    sp.exp((+t*x**2 - y**2)/(1+alpha+t**2))],
             't_var':t,
             'x_vars':[x,y],
             'alpha_var':alpha,
            },
            {'u':
                    [(x**3) + (y**2)*((t+0.5)**1.5) + x*y*alpha,
                     (x**2) + (y**3)*((t+0.5)**1.1) + x*y*alpha],
             't_var':t,
             'x_vars':[x,y],
             'alpha_var':alpha,
            }
         ]

def manufacture_elasticity_solution(u, x_vars, t_var, alpha_var=None, d1=2, d2=2, nu=0.25):
    '''Manufactures f from the elasticity equation given u (list of sympy equations)
    returns functions with unused dimensions=0'''
    print('manufacturing solution, from', d1, 'to', d2, 'dimensions.')
    assert d1>=d2
    assert d1<=2
    assert d2>=2
    epsilon_bar = np.array([u[0].diff(x_vars[0]), u[1].diff(x_vars[1]), u[1].diff(x_vars[0])+u[0].diff(x_vars[1])])
    C = np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])/(1-nu**2)
    sigma_bar = C @ epsilon_bar

    # static: f = -Div(sigma) (remember sigma != sigma_bar)
    f = [-sigma_bar[0].diff(x_vars[0]) -sigma_bar[2].diff(x_vars[1]),
         -sigma_bar[2].diff(x_vars[0]) -sigma_bar[1].diff(x_vars[1])]
    # add transient term; now f = u_tt - Div(sigma)
    f += np.array([u[0].diff(t_var,t_var), u[1].diff(t_var, t_var)])
    f_temp = lambdify([*[x_vars],t_var,alpha_var],f, "numpy")
    u_temp = lambdify([*[x_vars],t_var,alpha_var],u, "numpy")
    w_temp = lambdify([*[x_vars],t_var,alpha_var],[u[0].diff(t_var),u[1].diff(t_var)], "numpy")
    def f(x,t,alpha,time_delta=0):
        x = [x] if length(x) == 0 else x
        return f_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def u(x,t,alpha,time_delta=0):
        if length(x)>d2: # if x is a list, and not a single point, we unpack recursively
            return np.array([u(x_i,t,alpha,time_delta) for x_i in x])
        assert (length(x)) == (d2 if d2>1 else 0)
        x = [x] if length(x) == 0 else x
        return u_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    def w(x,t,alpha,time_delta=0):
        if length(x)>d2: # if x is a list, and not a single point, we unpack recursively
            return np.array([w(x_i,t,alpha,time_delta) for x_i in x])
        assert (length(x)) == (d2 if d2>1 else 0)
        x = [x] if length(x) == 0 else x
        return w_temp([*x, *[0 for i in range(d1-d2)]], t=t, alpha=alpha)
    return (f,u, w)
