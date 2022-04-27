# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:00:24 2020

@author: Aaron Hao
"""

import torch
import numpy as np
from gradient_functions import white_box_grad
from shuffle import ShuffleDataset
"""
def deepfool(image, model, grads, num_classes=10, overshoot=0.02, max_iter=50): ## fix thisss


       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image.requires_grad = True
    input_shape = image.shape
    pert_image = image
    
    prediction = model.predict(pert_image);
    copy = prediction.clone()
    f_image = copy.detach().cpu().numpy().flatten()
    I = (f_image).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]


    f_i = copy.detach().cpu().numpy().flatten() ## store prediction
    k_i = int(np.argmax(f_i)) ## current decision

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        gradients = grads(pert_image ,prediction)

        for k in range(1, num_classes): ## yo the order be wrong though

            # set new w_k and new f_k
            w_k = (gradients[k] - gradients[0]).detach().cpu().numpy()
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image
        
        pert_image = image.clone() + torch.from_numpy((1+overshoot)*r_tot)
        pert_image = pert_image.to(device=device, dtype=torch.float)
        loop_i += 1

        # compute new label
        prediction = model.predict(pert_image);
        copy = prediction.clone()
        f_i = copy.detach().cpu().numpy().flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, k_i, pert_image
"""

def deepfool(image, model, grads, num_classes=10, overshoot=0.02, max_iter=50): ## fix thisss

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
   """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image.requires_grad = True
    input_shape = image.shape
    pert_image = image
    
    prediction = model.predict(pert_image);
    copy = prediction.clone()
    f_image = copy.detach().cpu().numpy().flatten()
    I = (f_image).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]


    f_i = copy.detach().cpu().numpy().flatten() ## store prediction
    k_i = int(np.argmax(f_i)) ## current decision

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        gradients = grads(pert_image ,prediction)

        for k in range(1, num_classes): 

            # set new w_k and new f_k
            w_k = (gradients[k] - gradients[0]).detach().cpu().numpy()
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image
        
        pert_image = image.clone() + torch.from_numpy((1+overshoot)*r_tot)
        pert_image = pert_image.to(device=device, dtype=torch.float)
        loop_i += 1

        # compute new label
        prediction = model.predict(pert_image);
        copy = prediction.clone()
        f_i = copy.detach().cpu().numpy().flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, k_i, pert_image

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


def white_box_generate(model, dataset, grads= white_box_grad, delta=0.2, max_iter_uni = 6, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    v = torch.zeros_like(dataset[0][0])
    v = v.type(torch.float)
    fooling_rate = 0.0
    itr = 0
    num_images =  20 ###########check the dimensions for this one
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        
        print ('Starting pass number ', itr)
        
        for k in range(0, num_images):
            cur_img = dataset[k*itr][0]
            if int(np.argmax(model.predict(cur_img).detach().cpu().numpy())) == int(np.argmax(model.predict(cur_img+v).detach().cpu().numpy())):


                # Compute adversarial perturbation
                dr,iters,_,_ = deepfool(cur_img + v, model, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df) ## fix this
                    
                # Make sure it converged...
                if iters < max_iter_df-1:
                    temp = torch.from_numpy(dr).type(torch.float)
                    
                    v = v + temp

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)  
                    
        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = perturb_dataset(dataset, v)


        # Compute the estimated labels in batches
        print(v)
        numSuccess = 0            
        for i in range(num_images*100):
            if model.predict_label_no_query(dataset[i][0]) != model.predict_label_no_query(dataset_perturbed[i][0]):
                numSuccess+=1

        print('FOOLING RATE = ', numSuccess/(num_images*100))


    return v                    




def perturb_dataset(dataset, v):
    temp =[]
    for data in dataset:
        image = (data[0]+v).type(torch.float)
        temp.append((image, data[1]))
        
    return temp
    
    

        

        
