import wandb

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop


import onenow_gan_factory_optimizer as optimizer_factory



class Sequential_Network(object):
    
    """
    Sequential network can be used to instantiate generator or discriminator.
    It is capable of adapting its learning rate.
    """
    def __init__(self, net_name, net_sequence, optimizer_type, learning_rate,
                 loss_function, learning_metrics, model_path):
        self.net_name = net_name    
        self.net_sequence = net_sequence
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate 
        self.loss_function = loss_function
        self.learning_metrics = learning_metrics
        self.model_path = model_path
        self.neural_model = self.set_compiled_model(1, 0)

    def set_compiled_model(self, learning_fraction, training_step):
        adapted_learning_rate = self.learning_rate * learning_fraction
        wandb.log({'training_step': training_step, self.net_name +'learning_rate': adapted_learning_rate})

        # model: save
        try:
            self.neural_model.save_weights(self.get_path())
        except:
            pass

        # sequence
        sequence = Sequential()
        for net in self.net_sequence:
            net_model = net.get_model()
            net_model.summary()
            sequence.add(net_model)

        # optimizer    
        factory = optimizer_factory.Optimization_Algorithm()
        optimizer = factory.get_optimizer(self.optimizer_type, adapted_learning_rate)

        # compile
        sequence.compile(optimizer=optimizer, 
                         loss=self.loss_function, 
                         metrics=self.learning_metrics)

        # model: load
        try:
            sequence.load_weights(self.get_path())
        except:
            pass

        return sequence

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
