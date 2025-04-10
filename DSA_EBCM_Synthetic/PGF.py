import numpy as np

''' 
Poisson
'''
def psi_Poi(theta,para):
    mu=para[2]
    res=np.exp(mu*(theta-1))
    return res

def d1_psi_Poi(theta,para):
    mu=para[2]
    res=mu*np.exp(mu*(theta-1))
    return res
''' 
Regular
'''

def psi_Reg(theta,para):#,shift=3):
    a=para[2]
    res=theta**(a)
    return res

def d1_psi_Reg(theta,para):#,shift=3):
    a=para[2]
    res=a*theta**(a-1)
    return res 

''' 
Negative Binomial
'''
def psi_NBp(theta,para):
    r=para[2]
    shift=0
    p=para[-2]
    
    res=(theta**(shift))*(p/(1-(1-p)*theta))**r
    return res

def d1_psi_NBp(theta,para):
    r=para[2]
    shift=0
    p=para[-2]
    
    A=shift * (theta**(shift-1)) * (p/(1-(1-p)*theta))**r 
    B=(theta**shift)*(r*(1-p)*p**r)/((1-(1-p)*theta)**(1+r))
    res = A + B
    return res 


def psi_NB(theta,para):
    r=para[3]; p = para[3]/(para[2]+para[3])
    res=(p/(1-(1-p)*theta))**r
    return res

def d1_psi_NB(theta,para):
    r=para[3]; p = para[3]/(para[2]+para[3])
    res=(r*(1-p)*p**r)/((1-(1-p)*theta)**(1+r))
    return res 

def psi_SNB(theta,para):
    r=para[2]
    shift=para[3]
    p=para[-2]
    
    res=(theta**(shift))*(p/(1-(1-p)*theta))**r
    return res

def d1_psi_SNB(theta,para):
    r=para[2]
    shift=para[3]
    p=para[-2]
    
    A=shift * (theta**(shift-1)) * (p/(1-(1-p)*theta))**r 
    B=(theta**shift)*(r*(1-p)*p**r)/((1-(1-p)*theta)**(1+r))
    res = A + B
    return res 

def interp1d(obst, xmat):
    k = len(obst)
    jj = 0
    S_ti = np.zeros((k, xmat.shape[1]))
    S_ti[:, 0] = obst
    for i in range(k):
        for j in range(jj, xmat.shape[0]):
            if S_ti[i, 0] <= xmat[j, 0]:
                S_ti[i, 1:] = xmat[j, 1:]
                jj = j
                break
    return S_ti
