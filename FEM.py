'''This file contains functions used in the finite element method,
using lagrange polinomials as the basis function'''
import numpy as np
import quadrature
from matplotlib import pyplot as plt, cm
from utils import zero, length


def L2(pts, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    l=0
    def integrand(x): return (f1(x)-f2(x))**2
    for i in range(len(pts)-1):
        l+= quadrature.quadrature1d(pts[i],pts[i+1],5,integrand)
    return l**0.5

def L2_2d(pts,tri, f1, f2 = zero):
    '''returns L2 of f1-f2, (just f1 if f2 is empty)'''
    assert len(pts) == np.amax(tri)+1
    l=0
    def integrand(x):
        assert len(x) == 2
        return (f1(x)-f2(x))**2
    for triangle in tri:
        l+= quadrature.quadrature2d(pts[triangle[0]],pts[triangle[1]],pts[triangle[2]],4,integrand)
    return l**0.5

class Fem_1d:

    def __init__(self, pts, f, p, u_ex=None):
        self.Np = len(pts) #Np number of grid nodes
        self.pts = pts #triangulation, form [a0,a1,a2,a3,...aNp]
        self.f = f # source function
        self.p = p # degree of test functions'''
        self.u_ex = u_ex # exact function if provided
        assert (self.Np-1)%p==0

    def single_point_solution(self, x):
        pts=self.pts
        p=self.p
        vct = self.u_fem
        # we find the element where x is:
        for i in range(0,len(pts)-p,p): # pts[i] will be start of element
            if pts[i+p] >= x: # pts[i+p] is end of element
                result=0
                for j in range(p+1): # j local index of basis function
                    l = 1 # langrange pol
                    for k in range(p+1): # k
                        if k!=j:
                            l*=(x-pts[i+k])/(pts[i+j]-pts[i+k])
                    result += vct[i+j]*l
                return result
        print('Error: function evaluated outside domain')
        return 1/0

    def solution(self, x):
        if length(x)>0:
            return np.array([self.single_point_solution(x_i) for x_i in x])
        else:
            return self.single_point_solution(x)

    def relative_L2(self, ):
        return L2(self.pts,self.u_ex,self.solution)/L2(self.pts,self.u_ex)

    def set_u_ex(self, u_ex):
        self.u_ex = u_ex


class Poisson(Fem_1d):
    def __init__(self, pts, f, p=1, u_ex=None):
        super().__init__(pts,f,p,u_ex)

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
                                producti=1/(pts[k+i]-pts[k+m])
                                for l in range(p+1):
                                    if l!=i and l!=m:
                                        producti *= (x-pts[k+l])/(pts[k+i]-pts[k+l])
                                nphi += producti
                            if m!=j:
                                productj=1/(pts[k+j]-pts[k+m])
                                for l in range(p+1):
                                    if l!=j and l!=m:
                                        productj *= (x-pts[k+l])/(pts[k+j]-pts[k+l])
                                nphj += productj
                        return nphi*nphj
                    A[k+i,k+j] += quadrature.quadrature1d(pts[k],pts[k+p],p, integrand)
                def integrand(x):
                    # this is the integrand phi_i*f
                    phi = 1
                    for j in range(p+1):
                        if j!=i:
                            phi *= (x-pts[k+j])/(pts[k+i]-pts[k+j])
                    return phi*f(x)
                F[k+i] += quadrature.quadrature1d(pts[k],pts[k+p],5, integrand)
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
            g = [self.u_ex(self.pts[i]) for i in indices]
        for i in range(len(indices)):
            self.A[indices[i],:] = 0 # remove redundant equations
            self.A[indices[i],indices[i]] = -1
            self.F[indices[i]] = g[i]

    def add_Neumann_bdry(self, index,h):
        self.F[index] -= h

    def solve(self, ):
        self.u_fem=np.linalg.solve(-self.A,self.F)


class Heat(Fem_1d):
    def __init__(self, pts, f, p=1, u_ex=None, k=None):
        '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

        Np number of grid points
        pts: triangulation, form [a0,a1,a2,a3,...aNp]
        p: polynomial degree of test functoins
        f: source, form f(x,t)
        u_ex: exact solution. Form: u_ex(x,t=T) (important that T is default time,
                                    time is not provided when calculating L2)
        k: time step lenght. Only needs to be given when usin step() and not solve()
        '''
        super().__init__(pts,f,p,u_ex)
        self.time=0
        self.k = k

    def __discretize(self):
        Np = self.Np
        pts= self.pts
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
                                producti=1/(pts[k+i]-pts[k+m])
                                for l in range(p+1):
                                    if l!=i and l!=m:
                                        producti *= (x-pts[k+l])/(pts[k+i]-pts[k+l])
                                nphi += producti
                            if m!=j:
                                productj=1/(pts[k+j]-pts[k+m])
                                for l in range(p+1):
                                    if l!=j and l!=m:
                                        productj *= (x-pts[k+l])/(pts[k+j]-pts[k+l])
                                nphj += productj
                        return nphi*nphj
                    A[k+i,k+j] += quadrature.quadrature1d(pts[k],pts[k+p],p, integrand)

                    def integrand(x): # for m_ij
                        # this integrand is phi_i*phi_j
                        # formula: from rmnotes pg7
                        phi = 1 #phi_i
                        phj = 1 #phi_j
                        for m in range(p+1):
                            if m!=i:
                                phi*=(x-pts[k+m])/(pts[k+i]-pts[k+m])
                            if m!=j:
                                phj*=(x-pts[k+m])/(pts[k+j]-pts[k+m])
                        return phi*phj
                    M[k+i,k+j] += quadrature.quadrature1d(pts[k],pts[k+p],p+1, integrand)

                def integrand(x): # for f_i
                    # this is the integrand phi_i*f
                    phi = 1
                    for j in range(p+1):
                        if j!=i:
                            phi *= (x-pts[k+j])/(pts[k+i]-pts[k+j])
                    return phi*self.f(x=x, t=self.time)
                F[k+i] += quadrature.quadrature1d(pts[k],pts[k+p],5, integrand)

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
            g = [self.u_ex(self.pts[i], t=self.time) for i in indices]
        else:
            g = [gt(t=self.time) for gt in g]
        for i in range(len(indices)):
            self.MA[indices[i],:] = 0 # remove redundant equations
            self.M[indices[i],:] = 0
            self.MA[indices[i],indices[i]] = self.k
            self.F[indices[i]] = g[i]

    def step(self, g=None, correction = 0):
        '''Do one step of the backward euler scheme'''
        u_prev = self.u_fem
        self.time += self.k
        M,A,F = self.__discretize()
        self.F = F
        self.M = M
        self.MA = M+A*self.k
        self.__add_Dirichlet_bdry(g=g)
        self.u_fem = np.linalg.solve(self.MA, M@u_prev+self.F*self.k+correction) #Solve system


    def solve(self, time_steps, u0=None, g=None, T=1, callback=None):
        '''find u at t=T using backward euler'''
        assert self.time == 0 # this is only supposed to run on unsolved systems
        k = T/time_steps
        self.k=k

        if u0 != None:
            self.u_fem = u0(self.pts)
        else:
            self.u_fem = self.u_ex(self.pts, t=0)
        for t in range(time_steps):
            self.step(g)
            if callback!=None:
                callback(self.time,self.u_fem)


def in_triangle(p1,p2,p3,x):
    A = np.array([p2-p1, p3-p1]).T
    y = np.linalg.solve(A, x-p1)
    return y[0]>=-1e-15 and y[1]>=-1e-15 and y[0]+y[1]<=1+1e-15



class Fem_2d:

    def __init__(self, pts, tri, edge, f, p=1, u_ex=None):
        self.pts = pts # nodal points
        self.tri = tri #triangulation
        self.edge = edge # index of nodal points on the edge
        self.f = f # source function
        self.p = p # degree of test functions'''
        self.u_ex = u_ex # exact function if provided
        assert p==1 # p>1 not implemented (no plans to do so either)

    def single_point_solution(self, x):
        tri=self.tri
        pts=self.pts
        vct = self.u_fem
        # we find the element where x is:
        for i1,i2,i3 in tri:
            p1,p2,p3 = pts[i1],pts[i2],pts[i3]
            # Same as in_triangle(), but keep y
            A = np.array([p2-p1, p3-p1]).T
            x_local = np.linalg.solve(A, x-p1)
            if x_local[0]>=-1e-15 and x_local[1]>=-1e-15 and x_local[0]+x_local[1]<=1+1e-15:
                return vct[i1] + x_local[0]*(vct[i2]-vct[i1]) + x_local[1]*(vct[i3]-vct[i1])
        print('Error: function evaluated outside domain')
        return 1/0

    def solution(self, x):
        if length(x[0])>0: # not single x and y values
            return np.array([self.solution(x_i) for x_i in x])
        else:
            return self.single_point_solution(x)

    def relative_L2(self):
        return L2_2d(self.pts, self.tri,self.u_ex,self.solution)/L2_2d(self.pts, self.tri,self.u_ex)

    def set_u_ex(self, u_ex):
        self.u_ex = u_ex

    def plot_solution(self):

        N=int(len(self.pts)**0.5-1)*4

        X = np.array([np.linspace(-1,1,N+1)]*(N+1)).T
        Y = np.array([np.linspace(-1,1,N+1)]*(N+1))
        u_fem = self.solution(np.array([X.T,Y.T]).T)
        u_ex = self.u_ex(np.array([X.T,Y.T]).T)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1,projection='3d')
        ax2 = fig.add_subplot(1,2,2,projection='3d')
        ax1.plot_surface(X,Y,u_fem, cmap=cm.coolwarm)
        ax2.plot_surface(X,Y,u_ex, cmap=cm.coolwarm)
        #ax.view_init(elev=24, azim=-30)
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('u')
        plt.tight_layout()
        #plt.savefig(f'../preproject/1d_heat_figures/sols/{sol}.pdf')
        plt.show()

class Heat_2D(Fem_2d):
    def __init__(self, pts, tri, edge, f, p=1, u_ex=None, k=None):
        '''With backwards euler, so on step is (M+kA)u_new = u_old + kf

        pts: nodal points, form [[x1, y1], [x2, y2,] ...]
        tri: triangulation, indices of pts, from [[1,2,3], [0,3,4]... ]
               (where pts[1], pts[2], pts[3] makes up on triangle and so on)
        edge: list of indices of boundary nodes. Form [[0,1], [1,5]...]
               (where [pts[0],pts[1]] is a boundary edge and so on)
        p: polynomial degree of test functoins
        f: source, form f(x,t) where x = [x_1, x_2]
        u_ex: exact solution. Form: u_ex(x,t=T) (important that T is default time,
                                    time is not provided when calculating L2)
        k: time step lenght. Only needs to be given when usin step() and not solve()
        '''
        super().__init__(pts, tri, edge, f, p, u_ex)
        self.time=0
        self.k = k

    def __discretize(self):
        Np = len(self.pts)
        F = np.zeros((Np))
        A = np.zeros((Np,Np))
        M = np.zeros((Np,Np))

        for I in self.tri: # elementwise
            p0 = self.pts[I[0]] # points in the triangle
            p1 = self.pts[I[1]]
            p2 = self.pts[I[2]]
            # On the triangle phi_a is a+bx+cy such that phi_i(pj)=d_ij
            # We solve a linear system of equations to find a,b,c.
            m = np.array([
                [1, p0[0],p0[1]],
                [1, p1[0],p1[1]],
                [1, p2[0],p2[1]]
                ])
            phi = [0,0,0]
            phi[0] = np.linalg.solve(m, np.array([1,0,0]))
            phi[1] = np.linalg.solve(m, np.array([0,1,0]))
            phi[2] = np.linalg.solve(m, np.array([0,0,1]))
            # We loop through all the contributions to A
            for i in range(3):
                for j in range(3):
                    # The integrand for a_ij is the constant function nabla phi_i * nabla phi_j
                    def integrand_a(x):
                        return phi[i][1]*phi[j][1] + phi[i][2]*phi[j][2]
                    A[I[i],I[j]] += quadrature.quadrature2d(p0,p1,p2,3, integrand_a)

                    def integrand_m(x): # for m_ij, this integrand is phi_i*phi_j
                        return (phi[i][0] + phi[i][1]*x[0] + phi[i][2]*x[1])*(phi[j][0] + phi[j][1]*x[0] + phi[j][2]*x[1])
                    M[I[i],I[j]] += quadrature.quadrature2d(p0,p1,p2,4, integrand_m)

                def integrand_f(x): # for f_i, this is the integrand phi_i*f
                    return (phi[i][0] + phi[i][1]*x[0] + phi[i][2]*x[1])*self.f(x=x, t=self.time)
                F[I[i]] += quadrature.quadrature2d(p0,p1,p2,4, integrand_f)

        return M,A,F

    def __add_Dirichlet_bdry(self, g=None):
        '''g can be used instead of u_ex, either when u_ex is unknown or for testing with some other g'''
        assert g!=None or self.u_ex!=None
        for line in self.edge:
            for point in line: # this loops through every point twice, but ensures all points are considered
                self.MA[point,:] = 0 # remove redundant equations
                self.M[point,:] = 0
                self.MA[point,point] = self.k
                self.F[point] = self.u_ex(x=self.pts[point], t = self.time) if g==None else g[point](t=self.time)

    def step(self, g=None, correction = 0):
        '''Do one step of the backward euler scheme'''
        u_prev = self.u_fem
        self.time += self.k
        M,A,F = self.__discretize()
        self.F = F
        self.M = M
        self.MA = M+A*self.k
        self.__add_Dirichlet_bdry(g=g)
        self.u_fem = np.linalg.solve(self.MA, M@u_prev+self.F*self.k+correction) #Solve system


    def solve(self, time_steps, u0=None, g=None, T=1, callback=None):
        '''find u at t=T using backward euler'''
        assert self.time == 0 # this is only supposed to run on unsolved systems
        k = T/time_steps
        self.k=k

        if u0 != None:
            self.u_fem = u0(x=self.pts)
        else:
            self.u_fem = self.u_ex(x=self.pts, t=0)
        assert self.u_fem.shape == (len(self.pts),)
        for t in range(time_steps):
            self.step(g)
            if callback!=None:
                callback(self.time,self.u_fem)
