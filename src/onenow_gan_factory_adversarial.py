import wandb

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop


class Adversarial_Network(object):
    """
    Adversarial network contains a generator and a discriminator
    """   
    def __init__(self, generator_net, discriminator_net):
        self.discriminator_net = discriminator_net  
        self.generator_net = generator_net
    
    def get_G_model(self):
        sequence = self.generator_net.get_sequence() 
        first_in_sequence = sequence[0] 
        return first_in_sequence.get_model()
       
    # Getters and setters    
    def get_discriminator(self):
        return self.discriminator_net
    
    def get_discriminator_model(self):
        return self.discriminator_net.get_model()

    def get_generator(self):
        return self.generator_net

    def get_generator_model(self):
        return self.generator_net.get_model()    
    
    