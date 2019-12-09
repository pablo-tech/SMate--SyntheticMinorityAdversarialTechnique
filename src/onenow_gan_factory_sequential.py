import wandb
import time
import os

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

import onenow_gan_factory_optimizer as optimizer_factory



class Sequential_Network(object):
    
    """
    Sequential network can be used to instantiate generator or discriminator.
    It is capable of adapting its learning rate.
    """
    def __init__(self, net_name, net_sequence, optimizer_type, learning_rate,
                 loss_function, learning_metrics, model_path, 
                 transfer_path, num_frozen_layers = 6):
        self.net_name = net_name    
        self.net_sequence = net_sequence
        print("\n\n\n", "NETWORK_NAME=", net_name)
        i = 0
        for net in self.net_sequence:
            net.get_model().summary()
            ## plot network diagram
            plot_file = './' + self.net_name + str(i) + '.png'
            plot_model(net.get_model(), to_file=plot_file, show_shapes=True, show_layer_names=True)
            i+=1

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate 
        self.loss_function = loss_function
        self.learning_metrics = learning_metrics
        
        self.model_path = model_path
        self.transfer_path = transfer_path
        self.num_frozen_layers = num_frozen_layers
        self.neural_model = self.set_compiled_model(1, 0)
        
        
    def set_compiled_model(self, learning_fraction, training_step):
        
        chosen_path = self.get_path() + '.' + 'latest'  
        now_id = str(int(time.time()))
        
        adapted_learning_rate = self.learning_rate * learning_fraction
        wandb.log({'training_step': training_step, self.net_name +'learning_rate': adapted_learning_rate})

        ## model
        if not training_step == 0:
            try:
                self.save_weights(training_step, chosen_path)
                self.save_weights(training_step, self.get_path() + '.' + now_id)
            except:
                print("at_step", training_step, "unable_to_save_model=", self.get_path())
                pass

        ## sequence
        sequence = Sequential()
        i=0
        for net in self.net_sequence:
            net_model = net.get_model()
            # net_model.summary()
            # self.freeze_layers(net_model) 
            sequence.add(net_model)

        ## model: load
        try:                  
            if training_step == 0: 
                if not self.transfer_path == "":
                    chosen_path = self.transfer_path
                print("at_step", training_step, "TRANSFER_LEARNING_FROM=", chosen_path)
            sequence.load_weights(chosen_path)            
            print("at_step", training_step, "loaded_model=", chosen_path, "\n\n\n")
        except Exception as e:
            print(e)
            print("at_step", training_step, "unable_to_load_model=", chosen_path, "\n\n\n")
            pass
        
        ## freeze first layers: both generator and discriminator 
        if self.net_name == "GD_":
            self.freeze_layers(self.net_sequence[0].get_model()) # generator

        ## optimizer    
        factory = optimizer_factory.Optimization_Algorithm()
        optimizer = factory.get_optimizer(self.optimizer_type, adapted_learning_rate)

        # compile
        sequence.compile(optimizer=optimizer, 
                         loss=self.loss_function, 
                         metrics=self.learning_metrics)
            
        return sequence
    
    # https://stackoverflow.com/questions/46610732/how-to-freeze-some-layers-when-fine-tune-resnet50?rq=1
    def freeze_layers(self, sequence):
        for i in range(30): # unknown total number of layers
            try:
                layer = sequence.get_layer(index=i)
                if i < self.num_frozen_layers:
                    layer.trainable = False
                print(layer, "IS_TRAINABLE=", layer.trainable)    
            except:
                pass    
    
    def save_weights(self, step, path, isFreeze=False):
        # Then save    
        self.neural_model.save_weights(path)
        print("at_step", step, "saved_model=", path)
       

    """
    Getters and setters
    """
    def get_sequence(self):
        # "G_" or "GD_"
        return self.net_sequence
    
    def get_model(self):
        return self.neural_model
    
    def get_path(self):
        return self.model_path