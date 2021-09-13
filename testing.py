import numpy as np
import quadrature
import FEM
from matplotlib import pyplot as plt


def test1(p):

    # Define source function:
    def f(x):
        return 2*(1+x)**-3

    # Discretization:
    N = 13
    tri = np.linspace(0,1,N)
    #A, F = FEM.discretize_1d_poisson(N, tri, f)
    A, F = FEM.discretize_1d_poisson_v2(N, tri, f,p)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')

    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    ep=1e-10
    A[0,0] = 1/ep
    A[-1,-1] = 1/ep
    F[0] = 0
    F[-1] = 0
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')
    #print(A,F)

    u_fem = np.linalg.solve(-A, F) #Solve system

    def u_ex(x):
        return 1/(1+x)-1+x/2

    
    plt.plot(tri, u_fem, label='fem')
    #plt.plot(tri, u_ex(tri), label='u_ex')
    plt.plot(np.linspace(0,1,(N-1)*10+1), u_ex(np.linspace(0,1,(N-1)*10+1)), label='u_ex')
    plt.legend()
    plt.show()
    print(u_fem - u_ex(tri))

def test2(p=1):

    # Define source function:
    def f(x):
        return 6*x + np.pi**2/4*np.sin(np.pi*x/2)

    # Discretization:
    N = 13
    tri = np.linspace(0,1,N)
    A, F = FEM.discretize_1d_poisson_v2(N, tri, f, p)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')

    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    ep=1e-10
    A[0,0] = 1/ep
    A[-1,-1] = 1/ep
    F[0] = 0
    F[-1] = 0
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')
    #print(A,F)

    u_fem = np.linalg.solve(-A, F) #Solve system
    #print(u_fem)

    def u_ex(x):
        return x**3 - np.sin(np.pi/2*x)

    plt.plot(tri, u_fem, label='fem')
    plt.plot(np.linspace(0,1,(N-1)*10+1), u_ex(np.linspace(0,1,(N-1)*10+1)), label='u_ex')
    plt.legend()
    plt.show()
    print(u_fem - u_ex(tri))

def test3(N, p=1):

    # Define source function:
    def f(x):
        return 2*(1+x)**-3

    # Discretization:
    tri = np.linspace(0,1,N)
    A, F = FEM.discretize_1d_poisson_v2(N, tri, f, p)
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')

    # Add Dirichlet (zero) bdry conds (consider doing this in another way)
    ep=1e-10
    A[0,0] = 1/ep
    F[0] = 0
    h = 1/4 # outwards derivative of u at x=1
    F[-1] -= h
    #print('A is singular' if np.linalg.matrix_rank(A)<A.shape[0] else 'A is nonsingular')
    #print(A,F)

    u_fem = np.linalg.solve(-A, F) #Solve system
    #print(u_fem)

    def u_ex(x):
        return 1/(1+x)-1+x/2

    
    tri_fine = np.linspace(0,1,(N-1)*10+1)
    u_fem_fine = np.zeros(len(tri_fine))
    u_fem_fnc = FEM.fnc_from_vct(tri,u_fem,p)
    for i in range(len(tri_fine)):
        u_fem_fine[i] = u_fem_fnc(tri_fine[i])
    plt.plot(tri_fine, u_ex(tri_fine), label='u_ex')
    plt.plot(tri_fine, u_fem_fine, label='fem')
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
                A, F = FEM.discretize_1d_poisson_v2(N, tri, f, p)

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

        plt.loglog(ps,L2s.T, label=Ms)
        plt.xlabel('Polynomial degree')
        plt.grid()
        plt.legend(title='M:')
        plt.show()

def solve_heat(N, time_steps, u0, g, f, T=1):
    '''find u at t=T using backward euler'''
    k = T/time_steps
    tri = np.linspace(0,1,N+1)

    u_prev = u0(tri)
    for time_step in range(1,time_steps+1):
        M,A,F = FEM.discretize_1d_heat(N,tri,f(t=time_step*k))
        MA = M+A*k
        ep=1e-10
        MA[0,0]=k/ep
        MA[-1,-1]=k/ep
        F[0] = g(t=time_step*k)[0]/ep
        F[-1] = g(t=time_step*k)[1]/ep
        u_fem = np.linalg.solve(MA, M@u_prev+F*k) #Solve system
        u_prev = u_fem
    return u_fem, tri

def sindres_mfact_test(sol=0, alpha=0.5):
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

    if sol==5:
        alpha=1
        pi=np.pi
        sn=np.sin
        cs=np.cos
        a=alpha
        def f(t):
            def ft(x):
                return 2*pi*(cs(2*pi*t+a)-2*pi*sn(2*pi*t+a))*cs(2*pi*x) 
            return ft
        def u_ex(x,t=T): return 1+sn(2*pi*t+a)*cs(2*pi*x)

    if sol==3:
        def f(t):
            def ft(x):
                return alpha/(t+0.1)**2*(x*(1-x)+2*((x-1)*np.tanh(x/(t+0.1))-t-0.1))*np.cosh(x/(t+0.1))**-2
            return ft
        def u_ex(x,t=T): return 2+alpha*(x-1)*np.tanh(x/(t+0.1))

    if sol==4:
        def f(t):
            def ft(x):
                return 2*np.pi*(np.cos(2*np.pi*t+alpha)-2*np.pi*np.sin(2*np.pi*t+alpha))*np.cos(2*np.pi*x)
            return ft
        def u_ex(x,t=T): return 1 + np.sin(2*np.pi*t+alpha)*np.cos(2*np.pi*x)



    def u0(x): return u_ex(x,0)
    def g(t): return u_ex(0,t), u_ex(1,t)
    
    N=4
    time_steps=4
    L2s=[]
    dofs=[]
    for i in range(5):
        u_fem, tri = solve_heat(N,time_steps, u0, g, f, T)
        plt.plot(tri,u_fem, label=f'N,ts={N},{time_steps}')
        dofs.append(N*time_steps)
        N*=2
        time_steps*=2
        L2s.append(FEM.relative_L2(tri, u_ex, FEM.fnc_from_vct(tri,u_fem)))
    plt.plot(tri, u_ex(tri), label='u_ex')
    plt.legend()
    plt.show()

    plt.loglog(dofs,L2s)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    abdullah_bug_test()
    quit()
    test1(1)
    test1(2)
    test1(3)
    test1(4)
    test2()
    test2(2)
    test2(3)
    test2(4)
    test3(3)
    test3(6)
    test3(11)
    test3(21)
    test3(201)
    test3(3,2)
    test3(5,2)
    test3(11,2)
    test3(21,2)
    test3(201,2)
    test3(4,3)
    test3(7,3)
    test3(10,3)
    test3(22,3)
    test3(202,3)
    test3(5,4)
    test3(9,4)
    test3(13,4)
    test3(21,4)
    test3(201,4)
    sindres_mfact_test(4)
    sindres_mfact_test(5)
    sindres_mfact_test()
    sindres_mfact_test(1)
    sindres_mfact_test(2)
    sindres_mfact_test(3)
