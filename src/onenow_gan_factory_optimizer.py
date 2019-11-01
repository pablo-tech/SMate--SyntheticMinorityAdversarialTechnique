import wandb

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

 

# https://keras.io/optimizers/
class Optimization_Algorithm(object):
    """
    A desired optimizer is instantiated
    """
    def get_optimizer(self, optimizer_type, learning_rate):        
        if optimizer_type=='Adam':
            return self.get_adam(learning_rate)
        if optimizer_type=='RMSprop':
            return self.get_rms(learning_rate)        
    
    # https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L229
    def get_adam(self, learning_rate):
        return Adam(lr=learning_rate)

    def get_rms(self, learning_rate):
        # previous: decay=6e-8
        return RMSprop(lr=learning_rate)
    
    
    