#Transferring learning basing on the VGG-19w, working as the
#Neural Style Transfer #
# In[Basic preparation]
import os
import sys
import scipy.io
import scipy.misc
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
import time


#Content cost
def content_cost(aC, aG):
    
    #Take the dimension information
    m, n_H, n_W, n_C = aG.get_shape().as_list()
    
    #Lower the dimension for the image
    aC_unrolled = tf.reshape(aC, shape = [m, n_H*n_W, n_C])
    aG_unrolled = tf.reshape(aG, shape = [m, n_H*n_W, n_C])
    
    
    J_cont = tf.reduce_sum((aC_unrolled - aG_unrolled)**2)/(4*n_H*n_W*n_C)
    
    return J_cont



def style_cost_layer(aS, aG):
    
    m, n_H, n_W, n_C = aG.get_shape().as_list()
    
    aS = tf.transpose(tf.reshape(aS, shape = [n_H*n_W, n_C]))
    aG = tf.transpose(tf.reshape(aG, shape = [n_H*n_W, n_C]))
    
    GS = aS @ tf.transpose(aS)
    GG = aG @ tf.transpose(aG)
    
    J_style_layer = tf.reduce_sum(tf.square(GS - GG))/((2*n_H*n_W*n_C)**2)
    
    return J_style_layer
    
    
    
Layers_weight = [('conv1_1', 0.2), \
                 ('conv2_1', 0.2), \
                 ('conv3_1', 0.2), \
                 ('conv4_1', 0.2), \
                 ('conv5_1', 0.2)]



def style_cost_overall(model, Layers_weight):
    
    J_style = 0
    
    for name, coeff in Layers_weight:
        
        out = model[name]
        
        aS = sess.run(out)
        
        aG = out
        
        J_style_layer = style_cost_layer(aS, aG)
        
        J_style += coeff * J_style_layer
        
        
    return J_style
    
    
    
def Cost(J_cont, J_style, alpha, beta):
    
    J = alpha*J_cont + beta*J_style
    
    return J
    


tf.reset_default_graph()
sess = tf.InteractiveSession()
    
    
    
#Content image
content_image = cv2.imread("image_Content/Span.jpg")
content_image = content_image[...,::-1]
'''
plt.figure()
imshow(content_image)
'''
content_image = reshape_and_normalize_image(content_image)



#Style image
style_image = cv2.imread("image_Style/Moun_water_Style.jpg")
style_image = style_image[...,::-1]
'''
plt.figure()
imshow(style_image)
'''
style_image = reshape_and_normalize_image(style_image)



#Generated image initialization
init_gene_image = generate_noise_image(content_image)
'''
plt.figure()
imshow(init_gene_image[0])
'''


# In[Model deploy]
#pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
#pp.pprint(model)


#Content cost
sess.run(model['input'].assign(content_image))
out = model['conv4_2']

aC = sess.run(out)
aG = out
J_cont = content_cost(aC, aG)



#Style cost
sess.run(model['input'].assign(style_image))
J_style = style_cost_overall(model, Layers_weight)


#Overall cost
alpha = 10
beta  = 40 
J = Cost(J_cont, J_style, alpha, beta)


#Initialize the optimizer
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)



#Implementation
def model_NST(sess, input_image, num_iterations):
    
    sess.run(tf.global_variables_initializer())
    
    sess.run(model["input"].assign(input_image))
    
    
    for i in range(num_iterations):
        
        sess.run(train_step)
        
        image_G = sess.run(model["input"])
        
        
        if i%10 == 0:
            Jt, Jc, Js = sess.run([J, J_cont, J_style])
            print('')
            print('#############')
            print("Iteration " + str(i) + ":")
            print("Cont cost = " + str(Jc))
            print("Style cost = " + str(Js))
            print("Overal cost = " + str(Jt))
            print('')
            
            save_image("image_Generated/Iteration" + str(i) + ".jpg", image_G)
            
            
    save_image('image_Generated/Transferred_completed.jpg', image_G)
        
        
    return image_G


# In[Transferring]
tic = time.clock()
num_iterations = 400
model_NST(sess, init_gene_image, num_iterations)
toc = time.clock()
print('Time spending for %d iterations:' % num_iterations)
print('### ', (toc - tic), 'secs', ' ###')
      