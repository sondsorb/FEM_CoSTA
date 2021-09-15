import numpy as np

def sbmfact(T=1, alpha=0.5):
    return [
            (
                lambda x,t: 0,
                lambda x,t=T: alpha*(t+x*x/2)
            ),
            (
                lambda x,t: 1-alpha,
                lambda x,t=T: t+alpha*x*x/2
            ),
            (
                lambda x,t: 0.5/(t+alpha+1)**0.5-120*x*x-60*x+40,
                lambda x,t=T: (t+alpha+1)**0.5 + 10*x*x*(x-1)*(x+2)
            ),
            (
                lambda x,t: alpha/(t+0.1)**2*(x*(1-x)+2*((x-1)*np.tanh(x/(t+0.1))-t-0.1))*np.cosh(x/(t+0.1))**-2,
                lambda x,t=T: 2+alpha*(x-1)*np.tanh(x/(t+0.1))
            ),
            (
                lambda x,t: 2*np.pi*(np.cos(2*np.pi*t+alpha)+2*np.pi*np.sin(2*np.pi*t+alpha))*np.cos(2*np.pi*x) ,
                lambda x,t=T: 1+np.sin(2*np.pi*t+alpha)*np.cos(2*np.pi*x)
            )
    ]
