import numpy as np
import quadrature
import FEM
from matplotlib import pyplot as plt


def test1():

    # Define source function:
    def f(x):
        return 2*(1+x)**-3

    # Discretization:
    N = 2
    tri = np.linspace(0,1,N+1)
    A, F = FEM.discretize_1d_poisson(N, tri, f)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')

    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    ep=1e-10
    A[0,0] = 1/ep
    A[-1,-1] = 1/ep
    F[0] = 0
    F[-1] = 0
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')
    #print(A,F)

    u_fem = np.linalg.solve(A, F) #Solve system
    #print(u_fem)

    def u_ex(x):
        return 1/(1+x)-1+x/2

    
    plt.plot(tri, u_fem, label='fem')
    #plt.plot(tri, u_ex(tri), label='u_ex')
    plt.plot(np.linspace(0,1,N*10+1), u_ex(np.linspace(0,1,N*10+1)), label='u_ex')
    plt.legend()
    plt.show()
    print(u_fem - u_ex(tri))

def test2():

    # Define source function:
    def f(x):
        return 6*x + np.pi**2/4*np.sin(np.pi*x/2)

    # Discretization:
    N = 2
    tri = np.linspace(0,1,N+1)
    A, F = FEM.discretize_1d_poisson(N, tri, f)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')

    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    ep=1e-10
    A[0,0] = 1/ep
    A[-1,-1] = 1/ep
    F[0] = 0
    F[-1] = 0
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')
    #print(A,F)

    u_fem = np.linalg.solve(A, F) #Solve system
    #print(u_fem)

    def u_ex(x):
        return x**3 - np.sin(np.pi/2*x)

    plt.plot(tri, u_fem, label='fem')
    plt.plot(np.linspace(0,1,N*10+1), u_ex(np.linspace(0,1,N*10+1)), label='u_ex')
    plt.legend()
    plt.show()
    print(u_fem - u_ex(tri))

def test3(N):

    # Define source function:
    def f(x):
        return 2*(1+x)**-3

    # Discretization:
    tri = np.linspace(0,1,N+1)
    A, F = FEM.discretize_1d_poisson(N, tri, f)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')

    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    ep=1e-10
    A[0,0] = 1/ep
    F[0] = 0
    g=1/4
    def integrand_2(x):
        return g*(x-tri[-2])/(tri[-1]-tri[-2])
    F[-1] += quadrature.quadrature1d(tri[-2],tri[-1],4, integrand_2)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')
    #print(A,F)

    u_fem = np.linalg.solve(A, F) #Solve system
    #print(u_fem)

    def u_ex(x):
        return 1/(1+x)-1+x/2

    
    tri_fine = np.linspace(0,1,N*10+1)
    u_fem_fine = np.zeros(len(tri_fine))
    u_fem_fnc = FEM.fnc_from_vct(tri,u_fem)
    for i in range(len(tri_fine)):
        u_fem_fine[i] = u_fem_fnc(tri_fine[i])
    plt.plot(tri, u_fem, label='fem')
    #plt.plot(tri, u_ex(tri), label='u_ex')
    plt.plot(tri_fine, u_ex(tri_fine), label='u_ex')
    plt.plot(tri_fine, u_fem_fine, label='fem_fnc')
    plt.legend()
    plt.show()
    print(FEM.relative_L2(tri, u_ex, FEM.fnc_from_vct(tri,u_fem)))

def abdullah_bug_test():
    '''Taken from abdullahs preproject thesis chp 8, remember to credit'''
    '''Not implemented completely yet, only have linear elements so far'''#TODO
    for exponent in range(1,6):
        def u_ex(x): return x**exponent
        def f(x): 
            if exponent <2:
                return 0
            return exponent*(exponent-1)*x**(exponent-2)

        L2s=[]
        Ns = [1,2,4,8,16,32,64]
        for N in Ns:
            # Discretization:
            tri = np.linspace(0,1,N+1)
            A, F = FEM.discretize_1d_poisson(N, tri, f)
                    
            # Add Dirichlet (zero) bdry conds (consider doing this in another way)
            ep=1e-10
            A[0,0] = 1/ep
            F[0] = 0
            A[-1,-1] = 1/ep
            F[-1] = 1/ep # =g/ep, g=1

            u_fem = np.linalg.solve(A, F) #Solve system
            
            tri_fine = np.linspace(0,1,N*10+1)
            u_fem_fine = np.zeros(len(tri_fine))
            u_fem_fnc = FEM.fnc_from_vct(tri,u_fem)
            for i in range(len(tri_fine)):
                u_fem_fine[i] = u_fem_fnc(tri_fine[i])
            L2s.append(FEM.relative_L2(tri, u_ex, FEM.fnc_from_vct(tri,u_fem)))
        plt.loglog(Ns,L2s)
        plt.show()

def solve_heat(N, time_steps, u0, g, f, T=1):
    '''find u at t=T using backward euler'''
    k = T/time_steps
    tri = np.linspace(0,1,N+1)

    u_prev = u0(tri)
    for i in range(1,time_steps+1):
        M,A,F = FEM.discretize_1d_heat(N,tri,f(t=time_steps*k))
        MA = M+A*k
        ep=1e-10
        MA[0,0]=k/ep
        MA[-1,-1]=k/ep
        F[0] = g(t=time_steps*k)[0]/ep
        F[-1] = g(t=time_steps*k)[1]/ep
        u_fem = np.linalg.solve(MA, M@u_prev+F*k) #Solve system
        u_prev = u_fem
    return u_fem

def sindres_mfact_test(sol=0):
    '''testing equations from sindres paper.'''
    alpha = 1
    T=1

    if sol==0:
        def f(t):
            def ft(x):
                return 0
            return ft
        def u_ex(x,t=T): return alpha*(t+x*x/2)
    
    if sol==1:
        alpha = 1/2
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

    def u0(x): return u_ex(x,0)
    def g(t): return u_ex(0,t), u_ex(1,t)
    
    N=4
    time_steps=8
    for i in range(3):
        u_fem = solve_heat(N,time_steps, u0, g, f, T)
        plt.plot(np.linspace(0,1,N+1), u_fem, label=f'N,ts={N},{time_steps}')
        N*=2
        time_steps*=4
    plt.plot(np.linspace(0,1,N+1), u_ex(np.linspace(0,1,N+1)), label='u_ex')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #test1()
    #test2()
    sindres_mfact_test()
    sindres_mfact_test(1)
    sindres_mfact_test(2)
    quit()
    abdullah_bug_test()
    test3(2)
    test3(5)
    test3(10)
    test3(20)
    test3(200)
