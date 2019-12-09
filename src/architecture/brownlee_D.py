# TARGET: 32,32,3 CIFAR-10
# 
# REFERENCES
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

       
class DifferetiableNetwork(object):
    
    def __init__(self, creation_shape):
        self.net_config = self.get_new_config()
        self.net_config['creation_shape'] = creation_shape
        self.neural_net = self.get_new_model()
        
    def get_new_config(self):    
        config = {}
        return config
              
    def get_new_model(self):
        
        sequence = Sequential()
        
        in_shape = (32,32,3)
        # normal
        sequence.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
        sequence.add(LeakyReLU(alpha=0.2))
        # downsample
        sequence.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        # downsample
        sequence.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        # downsample
        sequence.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        # classifier
        sequence.add(Flatten())
        sequence.add(Dropout(rate=0.4))
        sequence.add(Dense(1, activation='sigmoid'))

        
        return sequence
                  
    def get_model(self):
        return self.neural_net
    
    def get_config(self):
        return self.net_config
