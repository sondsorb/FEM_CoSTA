'''This file contains functions used in the finite element method,
using lagrange polinomials as the basis function'''
import numpy as np
import quadrature

def length(x):
    try: 
        return len(x)
    except:
        return 0

def zero(x): return 0

def L2(tri, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    pass
    l=0
    def integrand(x): return (f1(x)-f2(x))**2
    for i in range(len(tri)-1):
        l+= quadrature.quadrature1d(tri[i],tri[i+1],4,integrand)
    return l**0.5


class Fem_1d:

    def __init__(tri, f, p):
        self.Np = len(tri) #Np number of grid nodes
        self.tri = tri #triangulation, form [a0,a1,a2,a3,...aNp]
        self.f = f # source function
        self.p = p # degree of test functions'''
        assert (Np-1)%p==0

    def single_point_solution(x): 
        tri=self.tri
        p=self.p
        vct = self.u_fem
        # we find the element where x is:
        for i in range(0,len(tri)-p,p): # tri[i] will be start of element
            if tri[i+p] >= x: # tri[i+p] is end of element
                result=0
                for j in range(p+1): # j local index of basis function
                    #print('r', result)
                    l = 1 # langrange pol
                    for k in range(p+1): # k 
                        if k!=j:
                            l*=(x-tri[i+k])/(tri[i+j]-tri[i+k])
                            #print('l',l)
                    result += vct[i+j]*l
                #print('r', result)
                return result
        print('Error: function evaluated outside domain')
        return 1/0
    
    def solution(x):
        if length(x)>0:
            return np.array([self.single_point_solution(x_i) for x_i in x])
        else:
            return single_point_solution(x)
    
    
    def relative_L2():
        return L2(self.tri,self.u_ex,self.solution)/L2(self.tri,self.u_ex)

class Poisson(Fem_1d):
    def __init__(tri, f, p=1):
        super.init(tri,f,p)
    
        F = np.zeros((Np))
        A = np.zeros((Np,Np))
    
        for k in range(0,Np-p,p): # elementwise: for element [a_k, a_k+p]
            # We loop through all the contributions to A
            for i in range(p+1):
                for j in range(p+1):
                    def integrand(x):
                        # this integrand is nabla(phi_i)*nabla(phi_j)
                        # formula: from rmnotes pg7
                        nphi = 0 #nabla(phi_i)
                        nphj = 0 #nabla(phi_j)
                        for m in range(p+1):
                            if m!=i:
                                producti=1/(tri[k+i]-tri[k+m])
                                for l in range(p+1):
                                    if l!=i and l!=m:
                                        producti *= (x-tri[k+l])/(tri[k+i]-tri[k+l])
                                nphi += producti
                            if m!=j:
                                productj=1/(tri[k+j]-tri[k+m])
                                for l in range(p+1):
                                    if l!=j and l!=m:
                                        productj *= (x-tri[k+l])/(tri[k+j]-tri[k+l])
                                nphj += productj
                        return nphi*nphj
                    A[k+i,k+j] += quadrature.quadrature1d(tri[k],tri[k+p],p, integrand)
                def integrand(x):
                    # this is the integrand phi_i*f
                    phi = 1
                    for j in range(p+1):
                        if j!=i:
                            phi *= (x-tri[k+j])/(tri[k+i]-tri[k+j])
                    return phi*f(x)
                F[k+i] += quadrature.quadrature1d(tri[k],tri[k+p],4, integrand)
            #print('a', integrand_1(tri[i])-1)
            #print('b', integrand_1(tri[i+1])-0)
            #print('c', integrand_2(tri[i])-0)
            #print('d', integrand_2(tri[i+1])-1)
            #assert integrand_1(tri[i])==1
            #assert integrand_1(tri[i+1])==0
            #assert integrand_2(tri[i])==0
            #assert integrand_2(tri[i+1])==1
    
        self.A=A
        self.F=F

    def solve():
        super.u_fem=np.linalg.solve(-A,F)

class heat(Fem_1d):
    def __init__(tri, f, p=1):
        '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

        Np number of grid points
        tri: triangulation, form [a0,a1,a2,a3,...aNp]'''
        super.init(tri,f,p)
    
        F = np.zeros((Np))
        A = np.zeros((Np,Np))
        M = np.zeros((Np,Np))

        for k in range(0,Np-p,p): # elementwise: for element [a_k, a_k+p]
            # We loop through all the contributions to A
            for i in range(p+1):
                for j in range(p+1):

                    def integrand(x): # for a_ij
                        # this integrand is nabla(phi_i)*nabla(phi_j)
                        # formula: from rmnotes pg7
                        nphi = 0 #nabla(phi_i)
                        nphj = 0 #nabla(phi_j)
                        for m in range(p+1):
                            if m!=i:
                                producti=1/(tri[k+i]-tri[k+m])
                                for l in range(p+1):
                                    if l!=i and l!=m:
                                        producti *= (x-tri[k+l])/(tri[k+i]-tri[k+l])
                                nphi += producti
                            if m!=j:
                                productj=1/(tri[k+j]-tri[k+m])
                                for l in range(p+1):
                                    if l!=j and l!=m:
                                        productj *= (x-tri[k+l])/(tri[k+j]-tri[k+l])
                                nphj += productj
                        return nphi*nphj
                    A[k+i,k+j] += quadrature.quadrature1d(tri[k],tri[k+p],p, integrand)

                    def integrand(x): # for m_ij
                        # this integrand is phi_i*phi_j
                        # formula: from rmnotes pg7
                        phi = 1 #phi_i
                        phj = 1 #phi_j
                        for m in range(p+1):
                            if m!=i:
                                phi*=(x-tri[k+m])/(tri[k+i]-tri[k+m])
                            if m!=j:
                                phj*=(x-tri[k+m])/(tri[k+j]-tri[k+m])
                        return phi*phj
                    M[k+i,k+j] += quadrature.quadrature1d(tri[k],tri[k+p],p+1, integrand)

                def integrand(x): # for f_i
                    # this is the integrand phi_i*f
                    phi = 1
                    for j in range(p+1):
                        if j!=i:
                            phi *= (x-tri[k+j])/(tri[k+i]-tri[k+j])
                    return phi*f(x)
                F[k+i] += quadrature.quadrature1d(tri[k],tri[k+p],4, integrand)

        self.M=M
        self.A=A
        self.F=F

    def solve(Ne, time_steps, u0, g, f, p, T=1):
        '''find u at t=T using backward euler'''
        k = T/time_steps
        tri = np.linspace(0,1,Ne*p+1)
    
        u_prev = u0(tri)
        for time_step in range(1,time_steps+1):
            M,A,F = discretize_1d_heat(tri,f(t=time_step*k),p)
            MA = M+A*k
            ep=1e-10
            MA[0,0]=k/ep
            MA[-1,-1]=k/ep
            F[0] = g(t=time_step*k)[0]/ep
            F[-1] = g(t=time_step*k)[1]/ep
            u_fem = np.linalg.solve(MA, M@u_prev+F*k) #Solve system
            u_prev = u_fem
        return u_fem



##########3 old, delete if classes work correctly:


def discretize_1d_poisson(Np, tri, f, p=1):
    '''Np number of grid nodes
    tri: triangulation, form [a0,a1,a2,a3,...aNp]
    f: source function
    p: degree of test functions'''
    assert (Np-1)%p==0

    F = np.zeros((Np))
    A = np.zeros((Np,Np))

    for k in range(0,Np-p,p): # elementwise: for element [a_k, a_k+p]
        # We loop through all the contributions to A
        for i in range(p+1):
            for j in range(p+1):
                def integrand(x):
                    # this integrand is nabla(phi_i)*nabla(phi_j)
                    # formula: from rmnotes pg7
                    nphi = 0 #nabla(phi_i)
                    nphj = 0 #nabla(phi_j)
                    for m in range(p+1):
                        if m!=i:
                            producti=1/(tri[k+i]-tri[k+m])
                            for l in range(p+1):
                                if l!=i and l!=m:
                                    producti *= (x-tri[k+l])/(tri[k+i]-tri[k+l])
                            nphi += producti
                        if m!=j:
                            productj=1/(tri[k+j]-tri[k+m])
                            for l in range(p+1):
                                if l!=j and l!=m:
                                    productj *= (x-tri[k+l])/(tri[k+j]-tri[k+l])
                            nphj += productj
                    return nphi*nphj
                A[k+i,k+j] += quadrature.quadrature1d(tri[k],tri[k+p],p, integrand)
            def integrand(x):
                # this is the integrand phi_i*f
                phi = 1
                for j in range(p+1):
                    if j!=i:
                        phi *= (x-tri[k+j])/(tri[k+i]-tri[k+j])
                return phi*f(x)
            F[k+i] += quadrature.quadrature1d(tri[k],tri[k+p],4, integrand)
        #print('a', integrand_1(tri[i])-1)
        #print('b', integrand_1(tri[i+1])-0)
        #print('c', integrand_2(tri[i])-0)
        #print('d', integrand_2(tri[i+1])-1)
        #assert integrand_1(tri[i])==1
        #assert integrand_1(tri[i+1])==0
        #assert integrand_2(tri[i])==0
        #assert integrand_2(tri[i+1])==1

    return A, F

def discretize_1d_heat(tri, f, p=1):
    '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

    Np number of grid points
    tri: triangulation, form [a0,a1,a2,a3,...aNp]'''
    Np = len(tri)

    F = np.zeros((Np))
    A = np.zeros((Np,Np))
    M = np.zeros((Np,Np))

    for k in range(0,Np-p,p): # elementwise: for element [a_k, a_k+p]
        # We loop through all the contributions to A
        for i in range(p+1):
            for j in range(p+1):

                def integrand(x): # for a_ij
                    # this integrand is nabla(phi_i)*nabla(phi_j)
                    # formula: from rmnotes pg7
                    nphi = 0 #nabla(phi_i)
                    nphj = 0 #nabla(phi_j)
                    for m in range(p+1):
                        if m!=i:
                            producti=1/(tri[k+i]-tri[k+m])
                            for l in range(p+1):
                                if l!=i and l!=m:
                                    producti *= (x-tri[k+l])/(tri[k+i]-tri[k+l])
                            nphi += producti
                        if m!=j:
                            productj=1/(tri[k+j]-tri[k+m])
                            for l in range(p+1):
                                if l!=j and l!=m:
                                    productj *= (x-tri[k+l])/(tri[k+j]-tri[k+l])
                            nphj += productj
                    return nphi*nphj
                A[k+i,k+j] += quadrature.quadrature1d(tri[k],tri[k+p],p, integrand)

                def integrand(x): # for m_ij
                    # this integrand is phi_i*phi_j
                    # formula: from rmnotes pg7
                    phi = 1 #phi_i
                    phj = 1 #phi_j
                    for m in range(p+1):
                        if m!=i:
                            phi*=(x-tri[k+m])/(tri[k+i]-tri[k+m])
                        if m!=j:
                            phj*=(x-tri[k+m])/(tri[k+j]-tri[k+m])
                    return phi*phj
                M[k+i,k+j] += quadrature.quadrature1d(tri[k],tri[k+p],p+1, integrand)

            def integrand(x): # for f_i
                # this is the integrand phi_i*f
                phi = 1
                for j in range(p+1):
                    if j!=i:
                        phi *= (x-tri[k+j])/(tri[k+i]-tri[k+j])
                return phi*f(x)
            F[k+i] += quadrature.quadrature1d(tri[k],tri[k+p],4, integrand)

    return M, A, F

def solve_heat(Ne, time_steps, u0, g, f, p, T=1):
    '''find u at t=T using backward euler'''
    k = T/time_steps
    tri = np.linspace(0,1,Ne*p+1)

    u_prev = u0(tri)
    for time_step in range(1,time_steps+1):
        M,A,F = discretize_1d_heat(tri,f(t=time_step*k),p)
        MA = M+A*k
        ep=1e-10
        MA[0,0]=k/ep
        MA[-1,-1]=k/ep
        F[0] = g(t=time_step*k)[0]/ep
        F[-1] = g(t=time_step*k)[1]/ep
        u_fem = np.linalg.solve(MA, M@u_prev+F*k) #Solve system
        u_prev = u_fem
    return u_fem

def length(x):
    try: 
        return len(x)
    except:
        return 0

def fnc_from_vct(tri, vct, p=1):
    '''returns fuction from vector of basis fuction coeffs.
    This could probably be made a lot faster, but that is not needed (yet at least)

    p polynimial degree'''
    def fnc(x):
        # we find the element where x is:
        for i in range(0,len(tri)-p,p): # tri[i] will be start of element
            if tri[i+p] >= x: # tri[i+p] is end of element
                result=0
                for j in range(p+1): # j local index of basis function
                    #print('r', result)
                    l = 1 # langrange pol
                    for k in range(p+1): # k 
                        if k!=j:
                            l*=(x-tri[i+k])/(tri[i+j]-tri[i+k])
                            #print('l',l)
                    result += vct[i+j]*l
                #print('r', result)
                return result
        print('Error: function evaluated outside domain')
        return 1/0

    def vector_fnc(x):
        if length(x)>0:
            return np.array([fnc(x_i) for x_i in x])
        else:
            return fnc(x)

    return vector_fnc


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
