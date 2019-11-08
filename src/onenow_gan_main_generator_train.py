#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Library: general
import time
import datetime
import math
import os
import logging


# In[2]:


# Library: randomness

import random
import scipy 
import numpy as np
import tensorflow as tf

"""
Result repdoducibility
"""
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


# In[3]:


print("SCIPY=", scipy.__version__)
print("NUMPY=", np.__version__)
print("TENSORFLOW=", tf.__version__)


# In[4]:


# Install dependencies 
import onenow_gan_install


# In[5]:


import wandb

get_ipython().system('wandb login 45396bf25753eeeb051e5567c5e7dd67446e3be4')


# In[6]:


from architecture import atienza_G as atienza_G
from architecture import atienza_D as atienza_D
from architecture import atienza_bertorello_G as atiber_G
from architecture import atienza_bertorello_D as atiber_D
from architecture import brownlee_G as brownlee_G
from architecture import brownlee_D as brownlee_D


def get_G(architecture_choice, input_n):
    if architecture_choice == 'atienza':
        return atienza_G.DifferetiableNetwork(input_n)
    if architecture_choice == 'atiber':
        return atiber_G.DifferetiableNetwork(input_n)
    if architecture_choice == 'brownlee':
        return brownlee_G.DifferetiableNetwork(input_n)
        
def get_D(architecture_choice, object_shape):    
    if architecture_choice == 'atienza':
        return atienza_D.DifferetiableNetwork(object_shape)
    if architecture_choice == 'atiber':
        return atiber_D.DifferetiableNetwork(object_shape)
    if architecture_choice == 'brownlee':
        return brownlee_D.DifferetiableNetwork(object_shape)


# In[7]:


# https://www.youtube.com/watch?v=XeQBsidyhWE
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy

# def CrossEntropyLoss(yHat, y):
#     if y == 1:
#       return -log(yHat)       // loss rapidly increasing with yHat -> 0
#     else:
#       return -log(1 - yHat)   // loss rapidly increasing with yHat -> 1

# In adversarial networks D is trained with binary_crossentropy:
# a) true x examples with y=1...yHat = D(x)
# b) fake G(z) examples with y=0...yHat = D(G(z))

# Thus D loss is defined combined heavyside:
# yTrue*log(D(x)) + (1-yFake)*log(1 − D(G(z)))

# And G is trained to minimize log(1 - D(G(z)))
# For faster training: D maximize log D(G(z)) has stronger gradients
# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618


# In[8]:


import onenow_gan_factory_adversarial as adversarial_factory
import onenow_gan_factory_sequential as sequential_factory

    
def get_adversarial_net(architecture_choice, optimizer_type, learning_rate, 
                        object_shape, G_input_n, net_config):
    """
    D is trained to: a) correctly label real example as 1, and b) correctly label fake example as 0
    Success for G succeeds fooling D, generating fake images indistiguishable from real ones
    Success for D is detecting fakes; ultimately accuracy converges to 50% (fakes as good as real) 
    """        
    # binary cross-entropy loss 
    discriminator_loss_function = 'binary_crossentropy' 
    discriminator_learning_metrics = ['accuracy']

    # optimizer: Generator
    generator_loss_function = 'binary_crossentropy'
    generator_learning_metrics = ['accuracy'] 

    G_net = get_G(architecture_choice, G_input_n) 
    D_net = get_D(architecture_choice, object_shape)
                    
    # model
    # TODO: rename Sequential_Network to NetworkModel
    discriminator_model = sequential_factory.Sequential_Network("D_", [D_net], optimizer_type, learning_rate, 
                discriminator_loss_function, discriminator_learning_metrics,
                net_config['discriminator_model_path'])
    
    # prevent discriminator from converging much faster than generator
    generator_learning_fraction = 2 # k TODO: turn into hyper param, use to inner loop instead
    generator_learning_rate = learning_rate / generator_learning_fraction
    
    generator_model = sequential_factory.Sequential_Network("GD_", [G_net, D_net], 
                                                            optimizer_type, generator_learning_rate, 
                                                            generator_loss_function, generator_learning_metrics,
                                                            net_config['generator_model_path'])

    return adversarial_factory.Adversarial_Network(generator_model, discriminator_model)
        


# In[9]:


import onenow_gan_utils as outil 

# https://www.tensorflow.org/api_docs/python/tf/keras/datasets
# https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/3-neural_network/autoencoder
def get_data_set(dataset_name, choice_class):
    
    # dataset
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()                

    # class 
    print("x_train=", x_train.shape)        
    if choice_class != "all":
        train_filter = []
        for i in range(len(x_train)):
            if y_train[i] == int(choice_class):
                train_filter.append(i)
        x_train = x_train[train_filter]            
        y_train = y_train[train_filter] 

    x_train = x_train.astype('float32')
    # scale from [0,255] to [-1,1]
    x_train = (x_train - 127.5) / 127.5
    print("x_train=", x_train.shape)
    return x_train, x_train[0].shape

train, shape = get_data_set("cifar10", "7")

# outil.plot_one_object(train[0], "save_me", "./")


# In[10]:


from math import sqrt

from onenow_gan_trainer import GAN_TRAINER


def run_train(dataset_name, data_class, architecture_choice, optimizer_type, learning_rate, net_config, logger):
    # config
    wandb.init(project=net_config['project_name'])
    wandb.config.update(net_config)
    # net
    x_train, object_shape = get_data_set(dataset_name, data_class)
    G_input_n = 100
    A_net = get_adversarial_net(architecture_choice, optimizer_type, learning_rate, 
                                object_shape, G_input_n, net_config)
    # train
    trainer = GAN_TRAINER(x_train, A_net, net_config, logger) 
    trainer.train(net_config['train_batch_size'], G_input_n)   


# In[ ]:


from onenow_gan_config import SystemConfig


if __name__ == '__main__':
    """
    Search for best model by iterating over hyper-parameters
    """
    
    # init
    version = "11"
    name = "gan-data"
    project_name = (name + "__v%s") % str(format(int(version), '03d'))
    global_config = SystemConfig(project_name)
    wandb_config = global_config.get_config()
    logger = logging.getLogger()    

    # define hyper param space
    hyperparam_space = {}  
    hyperparam_space['data_set'] = ['cifar10'] #  'mnist'    
    hyperparam_space['data_class'] = ['7'] #  'all'    
    hyperparam_space['architecture_choice'] = ['brownlee'] #  'atienza', 'atiber'
    hyperparam_space['optimizer_type'] = ['RMSprop']  # 'Adam', 
    hyperparam_space['learning_rate'] = [0.01*10**(-2)]  # 0.1*10**(-2), 0.001*10**(-2)
    hyperparam_space['batch_size'] = [128] # 4096, 8192, 1024, 256, 
    
    # iterate over hyper param space    
    for data_set in hyperparam_space['data_set']:
        for data_class in hyperparam_space['data_class']:
            for arch_choice in hyperparam_space['architecture_choice']:
                wandb_config['architecture_choice'] = arch_choice
                for optimizer_type in hyperparam_space['optimizer_type']: 
                    wandb_config['optimizer_type'] = optimizer_type
                    for learning_rate in hyperparam_space['learning_rate']:
                        wandb_config['learning_rate'] = learning_rate
                        for train_batch_size in hyperparam_space['batch_size']:
                            wandb_config['train_batch_size'] = train_batch_size
                            # path   
                            project_tag = '__data_set=' + data_set
                            project_tag += '__data_class=' + data_class
                            project_tag += '__architecture=' + arch_choice 
                            project_tag += '__optimizer=' + optimizer_type 
                            project_tag += '__learningrate='+ str(learning_rate) 
                            project_tag += '__batchsize=' + str(train_batch_size) 
                            wandb_config['project_tag'] = project_tag
                            print("project_name=" + project_name + "\t"+ "project_tag=" + project_tag)
                            # file
                            global_config.set_log_file(logger, project_tag)
                            global_config.set_creation_folder(project_tag)
                            global_config.set_model_folder(project_tag)
                            # run
                            try:
                                run_train(data_set, data_class,                                           arch_choice, optimizer_type, learning_rate,                                           wandb_config, logger)
                            except Exception as e:
                                print(e)

    
    # visualizer
    # https://lutzroeder.github.io/netron/
    
    # SHIFT Tab, ??
        
    # ASSERT shape
    
    # =========
    
        
    # any IMG dataset
    
    
    # any object dataset
        
    
    


# In[ ]:





# In[ ]:





# In[ ]:




