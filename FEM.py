'''This file contains functions used in the finite element method,
using lagrange polinomials as the basis function'''
import numpy as np
import quadrature

def discretize_1d_poisson(N, tri, f):
    '''N number of elements
    tri: triangulation, form [a0,a1,a2,a3,...aN]'''

    #TODO: implement choice of polynomial degree, and splnes an so in
    # Linear elements
    F = np.zeros((N+1))
    h = 1/N
    A = - np.diagflat(np.ones(N+1)*2)/h
    A = A + np.diagflat(np.ones(N)*(1),1)/h
    A = A + np.diagflat(np.ones(N)*(1),-1)/h
    #A[0,0] = 1
    #A[N,N] = 1


    for i in range(N): #elemntwise
        def integrand_1(x):
            return f(x)*(1-(x-tri[i])/(tri[i+1]-tri[i]))
        def integrand_2(x):
            return f(x)*(x-tri[i])/(tri[i+1]-tri[i])
        F[i] = F[i] + quadrature.quadrature1d(tri[i],tri[i+1],4, integrand_1)
        F[i+1] = F[i+1] + quadrature.quadrature1d(tri[i],tri[i+1],4, integrand_2)
        #print('a', integrand_1(tri[i])-1)
        #print('b', integrand_1(tri[i+1])-0)
        #print('c', integrand_2(tri[i])-0)
        #print('d', integrand_2(tri[i+1])-1)
        #assert integrand_1(tri[i])==1
        #assert integrand_1(tri[i+1])==0
        #assert integrand_2(tri[i])==0
        #assert integrand_2(tri[i+1])==1

    return A, F



def discretize_1d_heat(N, tri, f):
    '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

    N number of elements
    tri: triangulation, form [a0,a1,a2,a3,...aN]'''

    #TODO: implement choice of polynomial degree, and splnes an so in
    # Linear elements
    F = np.zeros((N+1))
    h = 1/N
    A = np.diagflat(np.ones(N+1)*2)/h
    A = A - np.diagflat(np.ones(N)*(1),1)/h
    A = A - np.diagflat(np.ones(N)*(1),-1)/h
    M = np.diagflat(np.ones(N+1)*2/3)*h
    M = M + np.diagflat(np.ones(N)*(1/6),1)*h
    M = M + np.diagflat(np.ones(N)*(1/6),-1)*h

    for i in range(N): #elemntwise
        def integrand_1(x):
            return f(x)*(1-(x-tri[i])/(tri[i+1]-tri[i]))
        def integrand_2(x):
            return f(x)*(x-tri[i])/(tri[i+1]-tri[i])
        F[i] = F[i] + quadrature.quadrature1d(tri[i],tri[i+1],4, integrand_1)
        F[i+1] = F[i+1] + quadrature.quadrature1d(tri[i],tri[i+1],4, integrand_2)

    return M, A, F

def fnc_from_vct(tri, vct, p=1):
    '''returns fuction from vector of basis fuction coeffs.
    This could probably be made a lot faster, but that is not needed (yet at least)

    p polynimial degree'''
    def fnc(x, tri=tri, vct=vct):
        # we find the element where x is:
        for i in range(0,len(tri)-p,p): # tri[i] will be start of element
            if tri[i+p] >= x: # tri[i+p] is end of element
                result=0
                for j in range(p+1): # j local index og basis function
                    l = 1 # langrange pol
                    for k in range(p+1): # k 
                        if k!=j:
                            l*=(x-tri[i+k])/(tri[i+j]-tri[i+k])
                    result += vct[i+j]*l
                return result
        print('Error: function evaluated outside domain')
        return 1/0
    return fnc


def relative_L2(tri,u_ex, u_fem):
    return L2(tri,u_ex,u_fem)/L2(tri,u_ex)

def zero(x): return 0

def L2(tri, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    pass
    l=0
    def integrand(x): return (f1(x)-f2(x))**2
    for i in range(len(tri)-1):
        l+= quadrature.quadrature1d(tri[i],tri[i+1],4,integrand)
    return l**0.5

