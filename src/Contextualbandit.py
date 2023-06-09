from observations import observational_process
from observations import binary_search_fronterior,binary_search_fronterior_A_x_z
import numpy as np
def h_func(Y,A,Z):
    return np.random.uniform()

def g_func(X,A):
    return np.random.uniform()
def CCB_IV(Z,A,X,Y,dataset, policy):
    # This policy is given by CCB-IV polivy
    # Use a contextual bandits with confounder(instrumental variable)
    # Use Rx be generated by Z,X,A and Rz caused by Z,X
    #这里可以先用一个比较naive的CCBmodel实现，之后再进行迭代。 
    # Set h1(Y,A,Z) as a projection to real number
    #第一步首先要做的是计算h1(Y,A,Z)
    policy=policy
    h1=h_func(Y,A,Z)
    fronterior=0
    frontierior, expectation=binary_search_fronterior(dataset,Z,policy,0,start_fronterior=0,end_fronterior=1, tol=1e-3)
    #print(frontierior)
    h11=fronterior+np.sum(Y)
    g=g_func(X,A)
    fronterior,expectation=binary_search_fronterior_A_x_z(dataset,Z,X,A,policy,0,start_fronterior=0,end_fronterior=1,tol=1e-3)
    g=h11+fronterior
    return g

def CCB_PV(a,u,x,o):
    #still to be done
    pass