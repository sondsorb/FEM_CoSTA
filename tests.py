# This file contains tests to verify that the code in this project works as it should

import numpy as np
import quadrature
import functions
import FEM
import solvers
from matplotlib import pyplot as plt

def test_reshaping():
    # Tests solvers.merge_first_dims
    i=3
    j=5
    k=7
    a = np.arange(i*j*k)
    b = np.reshape(a, (i,j,k))
    c = solvers.merge_first_dims(b)
    for ii in range(i):
        for jj in range(j):
            for kk in range(k):
                assert c[jj*i+ii,kk] == b[ii,jj,kk]
    print('reshape function passed')

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
        for Ne in [1,2,3,4,5]:
            Np=p*Ne+1
            e=test_function(u_ex, f, p, Np, True)
            assert e < e_prev/2
    print('test1 passed (FEM.Poisson, error -> 0 for increased p and Np, for dirichlet and neumann bdry cond)')

def test_function(u_ex, f, p, Np=13, neumann=False):
    model = FEM.Poisson(np.linspace(0,1,Np),f,p,u_ex)
    if not neumann:
        model.add_Dirichlet_bdry([0,-1])
    else:
        model.add_Dirichlet_bdry([0])
        model.add_Neumann_bdry([-1], 1/4)
    model.solve()
    return model.relative_L2()



def abdullah_bug_test():
    '''Taken from abdullahs preproject thesis chp 8, remember to credit'''
    '''Not implemented completely yet, only have linear elements so far'''#TODO
    fig1, axs1 = plt.subplots(3,2)
    fig2, axs2 = plt.subplots(3,2)
    for exponent in range(1,7):
        def u_ex(x): return x**exponent
        def f(x): 
            if exponent <2:
                return 0
            return exponent*(exponent-1)*x**(exponent-2)

        Nes = [2,4,8,16,32]#,64]#, 128, 256]
        ps = [1,2,3,4,5]
        L2s= np.zeros((len(Nes), len(ps)))
        for i in range(len(Nes)):
            for j in range(len(ps)):
                Ne=Nes[i]
                p=ps[j]
                Np = Ne*p+1

                # Discretization:
                model = FEM.Poisson(np.linspace(0,1,Np),f,p,u_ex)
                model.add_Dirichlet_bdry()
                model.solve ()
                L2s[i,j] = model.relative_L2()

                ##tri_fine = np.linspace(0,1,(Np-1)*10+1)
                ##u_fem_fnc = FEM.fnc_from_vct(tri,u_fem,p)
                ##plt.plot(tri_fine, u_ex(tri_fine), label='u_ex')
                ##plt.plot(tri_fine, u_fem_fnc(tri_fine), label='fem')
                ##plt.legend()
                ##plt.show()

        ##plt.loglog(Nes,L2s, label=ps)
        ##plt.xlabel('Number of elements')
        ##plt.grid()
        ##plt.legend(title='p:')
        ##plt.show()
        m = (exponent-1)//2
        n = (exponent-1)%2
        axs1[m,n].loglog(Nes,L2s, label=ps)
        axs1[m,n].set_xlabel('Number of elements')
        axs1[m,n].grid()
        axs1[m,n].legend(title='p:')

        ##plt.plot(ps,L2s.T, label=Nes)
        ##plt.yscale('log')
        ##plt.xlabel('Polynomial degree')
        ##plt.grid()
        ##plt.legend(title='Ne:')
        ##plt.show()
        axs2[m,n].plot(ps,L2s.T, label=Nes)
        axs2[m,n].set_yscale('log')
        axs2[m,n].set_xlabel('Polynomial degree')
        axs2[m,n].grid()
        axs2[m,n].legend(title='N:')
    plt.show()


def sindres_mfact_test(sol=0, alpha=0.5, p=4):
    '''testing equations from sindres paper.'''
    T=1
    solution = functions.Solution(T, f_raw=functions.SBMFACT[sol][0], u_raw=functions.SBMFACT[sol][1], zero_source=False, name=f'{sol}')
    solution.set_alpha(alpha)


    Ne=1
    time_steps=1
    L2s=[]
    dofs=[]
    for i in range(5):
        tri = np.linspace(0,1,Ne*p+1)
        model = FEM.Heat(tri, solution.f, p, solution.u)
        model.solve(time_steps, T=T)
        u_fem = model.u_fem
        tri_fine = np.linspace(0,1,10*Ne*p+1)

        plt.plot(tri_fine,model.solution(tri_fine), label=f'Ne,ts={Ne},{time_steps}')
        dofs.append((Ne*p-1)*time_steps)
        Ne*=2
        time_steps*=2
        L2s.append(model.relative_L2())
    plt.plot(tri_fine, solution.u(tri_fine), label='u_ex')
    plt.legend(title=f'sol {solution.name}')
    plt.show()

    plt.loglog(dofs,L2s)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test_reshaping()
    test1()
    #abdullah_bug_test()
    sindres_mfact_test(0,1)
    sindres_mfact_test(1,4)
    sindres_mfact_test(2,3)
    sindres_mfact_test(3,2)
    sindres_mfact_test(4,1)
    sindres_mfact_test(4,2)
    quit()
