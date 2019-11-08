# TARGET: 28,28,1 MNIST
# REFERENCES
# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# https://github.com/roatienza/Deep-Learning-Experiments
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop


# Batch normalization before RELU
# Decreasing number of filters, all same size
class DifferetiableNetwork(object):
    
    def __init__(self, input_n): 
        self.net_config = self.get_new_config()
        self.net_config['input_n'] = input_n
        self.neural_net = self.get_new_model()

    def get_new_config(self):
        config = {}
        config['filter_size'] = 5
        config['dropout'] = 0.4 # avoid generated images looking like noise
        config['momentum'] = 0.9
        config['dim'] = 7
        config['depth'] = 64+64+64+64
        config['volume'] = config['dim'] * config['dim'] * config['depth']
        config['reshape'] = (config['dim'], config['dim'], config['depth'])
        return config
        
    def get_new_model(self):
        sequence = Sequential()

        # input: fully connected
        sequence.add(Dense(self.net_config['volume'], input_dim=self.net_config['input_n'])) 
        sequence.add(BatchNormalization(momentum=self.net_config['momentum']))
        sequence.add(Activation('relu'))
        
        # upsample
        sequence.add(Reshape(self.net_config['reshape']))
        sequence.add(Dropout(self.net_config['dropout']))
        sequence.add(UpSampling2D())
        
        # hidden: convolutional
        sequence.add(Conv2DTranspose(filters = int(self.net_config['depth'] / 2), 
                                     kernel_size = self.net_config['filter_size'], 
                                     padding = 'same'),
                                     kernel_initializer = glorot_uniform(seed=0))
        sequence.add(BatchNormalization(momentum=self.net_config['momentum']))
        sequence.add(Activation('relu'))

        sequence.add(UpSampling2D()) 

        # hidden: convolutional
        sequence.add(Conv2DTranspose(filters = int(self.net_config['depth'] / 4), 
                                     kernel_size = self.net_config['filter_size'], 
                                     padding = 'same'),
                                     kernel_initializer = glorot_uniform(seed=0))
        sequence.add(BatchNormalization(momentum=self.net_config['momentum']))
        sequence.add(Activation('relu'))
        
        # hidden: convolutional
        sequence.add(Conv2DTranspose(filters = int(self.net_config['depth'] / 8), 
                                     kernel_size = self.net_config['filter_size'], 
                                     padding = 'same'),
                                     kernel_initializer = glorot_uniform(seed=0))
        sequence.add(BatchNormalization(momentum=self.net_config['momentum']))
        sequence.add(Activation('relu'))
        
        # hidden: convolutional
        sequence.add(Conv2DTranspose(filters = 1, 
                                     kernel_size = self.net_config['filter_size'], 
                                     padding = 'same'),
                                     kernel_initializer = glorot_uniform(seed=0))                     
        
        # output
        sequence.add(Activation('sigmoid'))
        
        return sequence
                                        
    def get_model(self):
        return self.neural_net
    
    def get_config(self):
        return self.net_config
        
