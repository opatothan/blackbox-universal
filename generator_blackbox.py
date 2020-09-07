# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:47:24 2020

@author: Paul Hao
"""
import torch
import numpy as np
import random 
import noise
from gradient_functions import white_box_grad
from shuffle import ShuffleDataset
from numpy import linalg as LA

def proj_lp(v, xi, p): ## what does this do???

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def createPerlinNoise(dimensions, res=6):
    """

    Parameters
    ----------
    dimensions : tuple or other iterable
        dimensions of the desired noise. For example, CIFAR-10 would be (3,32,32)
    res : TYPE
        how much detail/ how zoomed out the noise is. Higher means more detail. 

    Returns
    -------
    TYPE
        numpy array of noise.

    """
    if len(dimensions) ==3:
        sample = np.zeros(dimensions)
        x = dimensions[0]
        y = dimensions[1]
        z = dimensions[2]
        scale = [dimensions[0]/res,dimensions[1]/res, dimensions[2]/res]
        rand = random.randint(1,max(dimensions)*10)
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    sample[i][j][k] = noise.pnoise3((i+rand)/scale[0],(j+rand)/scale[1],(k+rand)/scale[2], octaves = 6,lacunarity = 2)
        
        return sample
    elif len(dimensions)==2:
        sample = np.zeros(dimensions)
        x = dimensions[0]
        y = dimensions[1]
        scale = [dimensions[0]/res,dimensions[1]/res]  
        rand = random.randint(1,max(dimensions)*10)
        for i in range(x):
            for j in range(y):
                sample[i][j] = noise.pnoise2((i+rand)/scale[0],(j+rand)/scale[1])
                
        return sample
    else:
        return -1


def black_box_generate(model, dataset, grads= white_box_grad, delta=0.2, max_iter_uni = 10, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    v = torch.zeros_like(dataset[0][0])
    rate_vs_itr = []
    v = v.type(torch.float)
    fooling_rate = 0.0
    itr = 0
    num_images =  20 ###########check the dimensions for this one
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        
        print ('Starting pass number ', itr)
        
        ##for k in range(0, num_images):
        for i in range(num_images):
            ##cur_img = dataset[k*itr][0]
            ##y0 = dataset[k*itr][1]
            cur_img = dataset[i][0]
            y0 = dataset[i][1]
            if int(np.argmax(model.predict(cur_img).detach().cpu().numpy())) == int(np.argmax(model.predict(cur_img+v).detach().cpu().numpy())):


                # Compute adversarial perturbation
                dr = attack_untargeted(model, cur_img+v, y0) ## fix this
                    
                # Make sure it converged...
                """
                if iters < max_iter_df-1:
                    temp = torch.from_numpy(dr).type(torch.float)
                    
                    v = v + temp

                    # Project on l_p ball
                    
                """
                v = v+ dr
                v = proj_lp(v, xi, p)  
                    
        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = perturb_dataset(dataset, v)


        # Compute the estimated labels in batches
        distortion = np.linalg.norm(v.numpy())
        print("Distortion: " + str(distortion))
        numSuccess = 0            
        for i in range(num_images*70):
            if model.predict_label_no_query(dataset[i][0]) != model.predict_label_no_query(dataset_perturbed[i][0]):
                numSuccess+=1
        fooling_rate =numSuccess/(num_images*70)
        rate_vs_itr.append((itr,fooling_rate))
        print('FOOLING RATE = ', fooling_rate)


    return v , rate_vs_itr                                




def perturb_dataset(dataset, v):
    temp =[]
    for data in dataset:
        image = (data[0]+v).type(torch.float)
        temp.append((image, data[1]))
        
    return temp

########################################################## signopt stuff ###########################################
    
def attack_untargeted(model, x0, y0, alpha = 0.2, beta = 0.001, iterations = 20, query_limit=20000,
                      distortion=None, seed=None, svm=False, momentum=0.0, stopping=0.0001):
    
    

    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """

    query_count = 0
    ls_total = 0
    
    if (model.predict_label(x0) != y0):
        return x0
    
    if seed is not None:
        np.random.seed(seed)

    # Calculate a good starting point.
    num_directions = 100
    best_theta, g_theta = None, float('inf')
    for i in range(num_directions):
        query_count += 1
        ###############################################################
        ##theta = createPerlinNoise((3,32,32), res = 5.5)
        ##################################################################
        theta = np.random.randn(*x0.shape)
        if model.predict_label(x0+torch.tensor(theta, dtype=torch.float))!=y0:
            initial_lbd = LA.norm(theta)
            theta /= initial_lbd
            lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
    

    if g_theta == float('inf'):
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0+torch.tensor(theta, dtype=torch.float))!=y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd 
    if g_theta == float('inf'):    

        return x0 


    


    # Begin Gradient Descent.
    xg, gg = best_theta, g_theta
    vg = np.zeros_like(xg)
    learning_rate = 1
    prev_obj = 100000
    distortions = [gg]

    
    
    
    
    for i in range(iterations):

        sign_gradient, grad_queries = sign_grad_v1(model, x0, y0, xg, original_theta=best_theta, initial_lbd=gg, h=beta)
      

        # Line lfarch
        ls_count = 0
        min_theta = xg
        min_g2 = gg
        min_vg = vg
        for _ in range(15):
            if momentum > 0:

#                     # Nesterov
#                     vg_prev = vg
#                     new_vg = momentum*vg - alpha*sign_gradient
#                     new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                new_vg = momentum*vg - alpha*sign_gradient
                new_theta = xg + new_vg
            else:
                
                #new_theta = xg - lr * m /np.sqrt(v_hat)
                #new_theta = xg+delta_adv
                new_theta = xg-alpha*sign_gradient
                
                
            new_theta /= LA.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(
                model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
            ls_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
                if momentum > 0:
                    min_vg = new_vg
            else:
                break

        if min_g2 >= gg: ## if failed, then try again with a smaller alpha
            for _ in range(15):
                alpha = alpha * 0.25
                if momentum > 0:

#                         # Nesterov
#                         vg_prev = vg
#                         new_vg = momentum*vg - alpha*sign_gradient
#                         new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                    new_vg = momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:#########################################################added implementation of previous gradient##################
                    new_theta = xg - alpha * sign_gradient
                    previous_gradient = sign_gradient
                    
                    #new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                if new_g2 < gg:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                    break
        if alpha < 1e-4:
            alpha = 1.0

            beta = beta*0.1
            if (beta < 1e-8):
                break
        
        xg, gg = min_theta, min_g2
        vg = min_vg
        
        query_count += (grad_queries + ls_count)
        ls_total += ls_count
        distortions.append(gg)

        if query_count > query_limit:
           break
        

        #if distortion is not None and gg < distortion:
        #    print("Success: required distortion reached")
        #    break

#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break            
#             prev_obj = gg
    target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float))



    #print(self.log)
    #print("Distortions: ", distortions)
    return torch.tensor(gg*xg, dtype=torch.float)
        
def sign_grad_v1(model, x0, y0, theta, initial_lbd, original_theta = 0, h=0.001, D=4, target=None):
    """
    Evaluate the sign of gradient by formulat
    sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
    """
    K = 200
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

    ########################################################################################
        ##u = createPerlinNoise((3,32,32), res = 6)
    ###################################################################################
        u = np.random.randn(*theta.shape)+ (original_theta*0.05)
        #u = Q[:,iii].reshape(sign_grad.shape)
        u /= LA.norm(u)
        
        sign = 1
        new_theta = theta + h*u
        new_theta /= LA.norm(new_theta)
        
        # Targeted case.
        if (target is not None and 
            model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float)) == target):
            sign = -1
            
        # Untargeted case
        preds.append(model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float)).item())
        if (target is None and
            model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float)) != y0):
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

def fine_grained_binary_search_local( model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
     
    if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float)) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float)) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float)) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float)) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search( model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best: 
        if model.predict_label(x0+torch.tensor(current_best*theta, dtype=torch.float)) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd
    
    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float)) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery
