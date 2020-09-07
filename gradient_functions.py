import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy.spatial
from scipy.linalg import qr
#from qpsolvers import solve_qp
import random


def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
    """
    Evaluate the sign of gradient by formulat
    sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
    """
    K = self.k
    sign_grad = np.zeros(theta.shape)
    queries = 0
    ### USe orthogonal transform
    #dim = np.prod(sign_grad.shape)
    #H = np.random.randn(dim, K)
    #Q, R = qr(H, mode='economic')
    preds = []
    for iii in range(K):
    #             # Code for reduced dimension gradient
    #             u = np.random.randn(N_d,N_d)
    #             u = u.repeat(D, axis=0).repeat(D, axis=1)
    #             u /= LA.norm(u)
    #             u = u.reshape([1,1,N,N])
        
        u = np.random.randn(*theta.shape)
        #u = Q[:,iii].reshape(sign_grad.shape)
        u /= LA.norm(u)
        
        sign = 1
        new_theta = theta + h*u
        new_theta /= LA.norm(new_theta)
        
        # Targeted case.
        if (target is not None and 
            self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == target):
            sign = -1
            
        # Untargeted case
        preds.append(self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()).item())
        if (target is None and
            self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) != y0):
            sign = -1
        queries += 1
        sign_grad += u*sign
    
    sign_grad /= K
    
    #         sign_grad_u = sign_grad/LA.norm(sign_grad)
    #         new_theta = theta + h*sign_grad_u
    #         new_theta /= LA.norm(new_theta)
    #         fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
    #         delta = (fxph - initial_lbd)/h
    #         queries += q1
    #         sign_grad *= 0.5*delta       
    
    return sign_grad, queries

def white_box_grad(x0, y0):
    output = torch.squeeze(y0)
    gradlist =[]
    for i in range(len(output)):
        gradlist.append(torch.autograd.grad(output[i], x0, retain_graph = True)[0]) ## hmmmmmmm take a look at this more later
        
        
    return gradlist
        
    
    