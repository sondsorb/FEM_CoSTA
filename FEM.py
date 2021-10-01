'''This file contains functions used in the finite element method,
using lagrange polinomials as the basis function'''
import numpy as np
import quadrature

def length(x):
    try: 
        return len(x)
    except:
        assert type(x) in [int, float, np.int64, np.float64]
        return 0

def zero(x, t=0): return 0

def L2(tri, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    pass
    l=0
    def integrand(x): return (f1(x)-f2(x))**2
    for i in range(len(tri)-1):
        l+= quadrature.quadrature1d(tri[i],tri[i+1],5,integrand)
        #l+= quadrature.quadrature1d(tri[i],tri[i]/2+tri[i+1]/2,5,integrand)
        #l+= quadrature.quadrature1d(tri[i]/2+tri[i+1]/2,tri[i+1],5,integrand)
    return l**0.5


class Fem_1d:

    def __init__(self, tri, f, p, u_ex=None):
        self.Np = len(tri) #Np number of grid nodes
        self.tri = tri #triangulation, form [a0,a1,a2,a3,...aNp]
        self.f = f # source function
        self.p = p # degree of test functions'''
        self.u_ex = u_ex # exact function if provided
        assert (self.Np-1)%p==0

    def single_point_solution(self, x): 
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
    
    def solution(self, x):
        if length(x)>0:
            return np.array([self.single_point_solution(x_i) for x_i in x])
        else:
            return self.single_point_solution(x)
    
    def relative_L2(self, ):
        return L2(self.tri,self.u_ex,self.solution)/L2(self.tri,self.u_ex)

    def set_u_ex(self, u_ex):
        self.u_ex = u_ex

    
class Poisson(Fem_1d):
    def __init__(self, tri, f, p=1, u_ex=None):
        super().__init__(tri,f,p,u_ex)

        Np=self.Np
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
                F[k+i] += quadrature.quadrature1d(tri[k],tri[k+p],5, integrand)
        self.A=A
        self.F=F

    def add_Dirichlet_bdry(self, indices=[0,-1], g=None):
        '''g is only needed if super does not have anu u_ex'''
        assert g!=None or self.u_ex!=None
        if length(indices)==0:
            indices = [indices]
            if g!=None:
                assert length(indices) == length(g)
                g = [g]
        if self.u_ex != None:
            g = [self.u_ex(self.tri[i]) for i in indices]
        ep=1e-16
        for i in range(len(indices)):
            self.A[indices[i],indices[i]] = -1/ep
            self.F[indices[i]] = g[i]/ep
    
    def add_Neumann_bdry(self, index,h):
        self.F[index] -= h

    def solve(self, ):
        self.u_fem=np.linalg.solve(-self.A,self.F)


class Heat(Fem_1d):
    def __init__(self, tri, f, p=1, u_ex=None, k=None):
        '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

        Np number of grid points
        tri: triangulation, form [a0,a1,a2,a3,...aNp]
        p: polynomial degree of test functoins
        f: source, form f(x,t)
        u_ex: exact solution. Form: u_ex(x,t=T) (important that T is default time, 
                                    time is not provided when calculating L2)
        k: time step lenght. Only needs to be given when usin step() and not solve()
        '''
        super().__init__(tri,f,p,u_ex)
        self.time=0
        self.k = k
    
    def __discretize(self):
        Np = self.Np
        tri= self.tri
        p = self.p
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
                    return phi*self.f(x=x, t=self.time)
                F[k+i] += quadrature.quadrature1d(tri[k],tri[k+p],5, integrand)

        return M,A,F

    def __add_Dirichlet_bdry(self, indices=[0,-1], g=None):
        '''g can be used instead of u_ex, either when u_ex is unknown or for testing with some other g'''
        assert g!=None or self.u_ex!=None
        if length(indices)==0:
            indices = [indices]
            if g!=None:
                assert length(indices) == length(g)
                g = [g]
        if g == None:
            g = [self.u_ex(self.tri[i], t=self.time) for i in indices]
        else:
            g = [gt(t=self.time) for gt in g]
        ep=1e-16
        for i in range(len(indices)):
            self.MA[indices[i],indices[i]] = self.k/ep
            self.F[indices[i]] = g[i]/ep

    def step(self, g=None, correction = 0):
        '''Do one step of the backward euler scheme'''
        u_prev = self.u_fem
        self.time += self.k
        M,A,F = self.__discretize()
        self.F = F
        self.MA = M+A*self.k
        self.__add_Dirichlet_bdry(g=g)
        self.u_fem = np.linalg.solve(self.MA, M@u_prev+F*self.k+correction) #Solve system


    def solve(self, time_steps, u0=None, g=None, T=1):
        '''find u at t=T using backward euler'''
        assert self.time == 0 # this is only supposed to run on unsolved systems
        k = T/time_steps
        self.k=k
    
        if u0 != None:
            self.u_fem = u0(self.tri)
        else:
            self.u_fem = self.u_ex(self.tri, t=0)
        for t in range(time_steps):
            self.step(g)
