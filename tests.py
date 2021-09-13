# This file contains test to verify that the code in this project works as it should

import numpy as np
import quadrature
import FEM
from matplotlib import pyplot as plt

def test1():
    def f(x): return 2*(1+x)**-3
    def u_ex(x): return 1/(1+x)-1+x/2
    e_prev=10
    for p in [1,2,3,4]:
        e=test_function(u_ex, f, p)
        # havent thought about the convergence rate here,
        # so just check if -> 0
        assert e < e_prev/2 

    def f(x): return 6*x + np.pi**2/4*np.sin(np.pi*x/2)
    def u_ex(x): return x**3 - np.sin(np.pi/2*x)
    e_prev=10
    for p in [1,2,3,4]:
        e=test_function(u_ex, f, p)
        assert e < e_prev/2

    def u_ex(x): return 1/(1+x)-1+x/2
    def f(x): return 2*(1+x)**-3
    for p in [1,2,3,4,5]:
        e_prev=10
        for M in [1,2,3,4,5]:
            N=p*M+1
            e=test_function(u_ex, f, p, N, True)
            assert e < e_prev/2
    print('test1 passed (error -> 0 for increased p and N, for dirichlet and neumann bdry cond)')


def test_function(u_ex, f, p, N=13, neumann=False):
    # Discretization:
    tri = np.linspace(0,1,N)
    A, F = FEM.discretize_1d_poisson(N, tri, f,p)
    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    if not neumann:
        ep=1e-10
        A[0,0] = 1/ep
        A[-1,-1] = 1/ep
        F[0] = 0
        F[-1] = 0
    if neumann:
        # I hardcoded the neumann conditions for the fuction u=1/(1+x)-1+x/2
        ep=1e-10
        A[0,0] = 1/ep
        F[0] = 0
        h = 1/4 # outwards derivative of u at x=1
        F[-1] -= h

    # Solve system
    u_fem = np.linalg.solve(-A, F)

    #plt.plot(tri, u_fem, label='fem')
    ##plt.plot(tri, u_ex(tri), label='u_ex')
    #plt.plot(np.linspace(0,1,(N-1)*10+1), u_ex(np.linspace(0,1,(N-1)*10+1)), label='u_ex')
    #plt.legend()
    #plt.show()
    tri_fine=np.linspace(0,1,(N-1)*10+1)
    return FEM.relative_L2(tri_fine, u_ex, FEM.fnc_from_vct(tri,u_fem,p))



def abdullah_bug_test():
    '''Taken from abdullahs preproject thesis chp 8, remember to credit'''
    '''Not implemented completely yet, only have linear elements so far'''#TODO
    for exponent in range(1,6):
        def u_ex(x): return x**exponent
        def f(x): 
            if exponent <2:
                return 0
            return exponent*(exponent-1)*x**(exponent-2)

        Ms = [2,4,8,16,32,64]
        ps = [1,2,3,4]
        L2s= np.zeros((len(Ms), len(ps)))
        for i in range(len(Ms)):
            for j in range(len(ps)):
                M=Ms[i]
                p=ps[j]

                # Discretization:
                N = M*p+1
                tri = np.linspace(0,1,N)
                A, F = FEM.discretize_1d_poisson(N, tri, f, p)

                # Add Dirichlet (zero) bdry conds (consider doing this in another way)
                ep=1e-10
                A[0,0] = -1/ep
                F[0] = 0
                A[-1,-1] = -1/ep
                F[-1] = 1/ep # this is =g/ep, g=1

                u_fem = np.linalg.solve(-A, F) #Solve system

                #tri_fine = np.linspace(0,1,(N-1)*10+1)
                #u_fem_fine = np.zeros(len(tri_fine))
                #u_fem_fnc = FEM.fnc_from_vct(tri,u_fem,p)
                #for k in range(len(tri_fine)):
                #    u_fem_fine[k] = u_fem_fnc(tri_fine[k])
                #plt.plot(tri_fine, u_ex(tri_fine), label='u_ex')
                #plt.plot(tri_fine, u_fem_fine, label='fem')
                #plt.legend()
                #plt.show()

                u_fem_fnc = FEM.fnc_from_vct(tri,u_fem,p)
                L2s[i,j] = FEM.relative_L2(tri, u_ex, FEM.fnc_from_vct(tri,u_fem,p))
        plt.loglog(Ms,L2s, label=ps)
        plt.xlabel('Number of elements')
        plt.grid()
        plt.legend(title='p:')
        plt.show()

        plt.plot(ps,L2s.T, label=Ms)
        plt.yscale('log')
        plt.xlabel('Polynomial degree')
        plt.grid()
        plt.legend(title='M:')
        plt.show()


def sindres_mfact_test(sol=0, alpha=0.5, p=4):
    '''testing equations from sindres paper.'''
    T=1

    if sol==0:
        def f(t):
            def ft(x):
                return 0
            return ft
        def u_ex(x,t=T): return alpha*(t+x*x/2)
    
    if sol==1:
        def f(t):
            def ft(x):
                return 1-alpha
            return ft
        def u_ex(x,t=T): return t+alpha*x*x/2
    
    if sol==2:
        def f(t):
            def ft(x):
                return 0.5/(t+alpha+1)**0.5-120*x*x-60*x+40
            return ft
        def u_ex(x,t=T): return (t+alpha+1)**0.5 + 10*x*x*(x-1)*(x+2)

    if sol==3:
        def f(t):
            def ft(x):
                return alpha/(t+0.1)**2*(x*(1-x)+2*((x-1)*np.tanh(x/(t+0.1))-t-0.1))*np.cosh(x/(t+0.1))**-2
            return ft
        def u_ex(x,t=T): return 2+alpha*(x-1)*np.tanh(x/(t+0.1))

    if sol==4:
        # abrreviatiosn
        pi=np.pi
        sn=np.sin
        cs=np.cos
        a=alpha
        def f(t):
            def ft(x):
                return 2*pi*(cs(2*pi*t+a)+2*pi*sn(2*pi*t+a))*cs(2*pi*x) 
            return ft
        def u_ex(x,t=T): return 1+sn(2*pi*t+a)*cs(2*pi*x)


    def u0(x): return u_ex(x,0)
    def g(t): return u_ex(0,t), u_ex(1,t)

    M=1
    time_steps=1
    L2s=[]
    dofs=[]
    for i in range(5):
        u_fem = FEM.solve_heat(M,time_steps, u0, g, f, p, T)
        tri = np.linspace(0,1,M*p+1)
        tri_fine = np.linspace(0,1,10*M*p+1)

        u_fem_fnc = FEM.fnc_from_vct(tri,u_fem,p)
        
        plt.plot(tri_fine,u_fem_fnc(tri_fine), label=f'M,ts={M},{time_steps}')
        dofs.append((M*p-1)*time_steps)
        M*=2
        time_steps*=2
        L2s.append(FEM.relative_L2(tri, u_ex, FEM.fnc_from_vct(tri,u_fem)))
    plt.plot(tri_fine, u_ex(tri_fine), label='u_ex')
    plt.legend(title=f'sol {sol}')
    plt.show()

    plt.loglog(dofs,L2s)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test1()
    sindres_mfact_test(0,1)
    sindres_mfact_test(1,4)
    sindres_mfact_test(2,3)
    sindres_mfact_test(3,2)
    sindres_mfact_test(4,1)
    abdullah_bug_test()
    quit()
