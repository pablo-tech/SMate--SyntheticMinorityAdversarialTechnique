import numpy as np
from matplotlib import pyplot as plt

import wandb

import onenow_gan_utils as outil


# TODO: avoid Discriminator overfitting: alternate between k steps of optimizing D and one step of optimizing G.
# As long as G changes slowly enough, D is maintained near its optimal solution
class GAN_TRAINER(object):
    """
    Train a Generative Adversarial Network, where Discriminator and Generator play a game.
    Discriminator is the D network.  Generator is the G network.
    D is modeled on its output, G is trained on D's output (GD concatenated)
    G takes input z, and outputs G(z), fakes to fool D.
    Every batch, G and D networks are trained simultaneously.
    D is trained on 1/2 fake examples from G, and 1/2 true examples from the training set.
    """
    def __init__(self, x_train, general_adversarial_network, net_config, global_logger): 
        self.x_train = x_train
        self.g_a_n = general_adversarial_network
        self.net_config = net_config
        self.logger = global_logger
        
    def train(self, batch_size, G_input_n):                
        # init
        last_kpi_value = 1 
        last_kpi_percent_momentum = 1         
        i = 1
        
        while True:                       
            # train
            x_true_objects = self.get_true_objects(batch_size)
            discriminator_metrics = self.train_discriminator(x_true_objects, batch_size, G_input_n)
            generator_metrics = self.train_generator(batch_size, G_input_n)
            # metrics
            kpi_value = self.get_kpi_value(generator_metrics, discriminator_metrics)
            kpi_percent_momentum = self.get_kpi_momentum(kpi_value, last_kpi_value, last_kpi_percent_momentum)
            # log
            self.print_log(i, generator_metrics, discriminator_metrics, kpi_percent_momentum)    
            self.plot_step_objects(i, G_input_n)
            # adaptation and convergence 
            self.adapt_learning(i, kpi_percent_momentum)
            self.check_convergence(i, kpi_percent_momentum)
            # housekeeping
            last_kpi_value = kpi_value
            last_kpi_percent_momentum = kpi_percent_momentum
            i += 1
    
    """
    training methods
    """          
    # Discriminator(x_true_objects)=1, and Discriminator(Generator(z))=0  
    def train_discriminator(self, x_true_objects, batch_size, G_input_n):
        _, G_of_z_fake_objects = self.get_fake_objects(batch_size, G_input_n)
        x_true_plus_fake = np.concatenate((x_true_objects, G_of_z_fake_objects))
        y_true_plus_fake = np.ones([2*batch_size, 1])    # true=1
        y_true_plus_fake[batch_size:, :] = 0             # fake=0
        discriminator_metrics = self.g_a_n.get_discriminator_model(). \
                                     train_on_batch(x_true_plus_fake, y_true_plus_fake)
        return discriminator_metrics
    
    # Generator seeks to make Discriminator(Generaator(z))=1 (appear true)
    def train_generator(self, batch_size, G_input_n):
        z_noise, _ = self.get_fake_objects(batch_size, G_input_n)
        y_fake_pass_as_true = np.ones([batch_size, 1])      
        generator_metrics = self.g_a_n.get_generator_model().train_on_batch(z_noise, y_fake_pass_as_true)
        return generator_metrics
                        
    def adapt_learning(self, step_count, kpi_percent_momentum):
        if self.is_check_point(step_count): 
            learning_rate_factor = kpi_percent_momentum
            self.g_a_n.get_generator().set_compiled_model(learning_rate_factor, step_count)
            self.g_a_n.get_discriminator().set_compiled_model(learning_rate_factor, step_count)        
                            
    
    """
    metric methods
    """    
    def network_metrics(self, generator_metrics, discriminator_metrics):
        generator_loss = generator_metrics[0]
        generator_accuracy = generator_metrics[1]
        discriminator_loss = discriminator_metrics[0]
        discriminator_accuracy = discriminator_metrics[1]
        return generator_loss, generator_accuracy, discriminator_loss, discriminator_accuracy             
    
    def get_kpi_value(self, generator_metrics, discriminator_metrics):
        _, _, _, discriminator_accuracy = self.network_metrics(generator_metrics, discriminator_metrics)
        return discriminator_accuracy

    def get_kpi_momentum(self, kpi_value, last_kpi_value, last_kpi_percent_momentum):  
        kpi_momentum_beta = 0.99
        kpi_change_percent = abs((last_kpi_value - kpi_value) / kpi_value)
        kpi_percent_momentum = kpi_momentum_beta * last_kpi_percent_momentum + \
                                   (1-kpi_momentum_beta) * kpi_change_percent
        return kpi_percent_momentum

    """
    data set methods
    """
    def get_true_objects(self, batch_size):
        x_true_index = np.random.randint(0, self.x_train.shape[0], size=batch_size)
        x_true_objects = self.x_train[x_true_index, :, :, :]
        return x_true_objects
    
    # G_of_z_fake_objects, z_noise
    def get_fake_objects(self, batch_size, G_input_n):
        z_noise = self.draw_G_random_input_noise(batch_size, G_input_n)
        G_of_z_fake_objects = self.g_a_n.get_G_model().predict(z_noise)
        return z_noise, G_of_z_fake_objects

    def draw_G_random_input_noise(self, num_samples, G_input_n):
        return np.random.uniform(-1.0, 1.0, size=[num_samples, G_input_n])

     
    """
    print/plot methods
    """    
    def print_log(self, step_count, generator_metrics, discriminator_metrics, kpi_percent_momentum):
        
        generator_loss, generator_accuracy, discriminator_loss, discriminator_accuracy = \
                self.network_metrics(generator_metrics, discriminator_metrics) 
        
        if self.is_check_point(step_count): 
            wandb.log({'training_step': step_count, 'discriminator_loss': discriminator_loss})
            wandb.log({'training_step': step_count, 'fake+real_catch_accuracy': discriminator_accuracy})
            wandb.log({'training_step': step_count, 'generator_loss': generator_loss})
            wandb.log({'training_step': step_count, 'fake_catch_accuracy': generator_accuracy})
            wandb.log({'training_step': step_count, 'kpi_percent_momentum': kpi_percent_momentum})
            self.logger.info("GENERATOR_%d: loss=%f, accuracy=%f" % (step_count, generator_loss, generator_accuracy))
            self.logger.info("DISCRIMINATOR_%d loss=%f, accuracy=%f" % (step_count, discriminator_loss, discriminator_accuracy))

    def plot_step_objects(self, step_count, G_input_n):
        if self.is_check_point(step_count): 
            batch_size = 16 # subplot 4x4
            self.plot_step_fake_objects(step_count, G_input_n, batch_size)
            self.plot_step_real_objects(step_count, G_input_n, batch_size)

    def plot_step_fake_objects(self, step_count, G_input_n, batch_size):
        _, objects = self.get_fake_objects(batch_size, G_input_n)
        # latest
        latest_filename = self.net_config['project_name'] + self.net_config['project_tag'] + "_latest=fake.png"
        num_rows = self.x_train[0].shape[0]
        num_cols = self.x_train[0].shape[1]
        outil.plot_objects(num_rows, num_cols, \
                          objects, latest_filename, self.net_config['creation_root'])      
        # all
        full_filename = self.net_config['project_name'] + self.net_config['project_tag'] + \
                   "_%s.png" % str(format(int(step_count), '07d'))
        outil.plot_objects(num_rows, num_cols, \
                          objects, full_filename, self.net_config['creation_full_path'])      
            
    def plot_step_real_objects(self, step_count, G_input_n, batch_size):
        objects = self.get_true_objects(batch_size)
        latest_filename = self.net_config['project_name'] + self.net_config['project_tag'] + "_latest=real.png"
        num_rows = self.x_train[0].shape[0]
        num_cols = self.x_train[0].shape[1]
        outil.plot_objects(num_rows, num_cols, \
                  objects, latest_filename, self.net_config['creation_root'])      


    """
    milestone/convergence methods
    """    
    def is_check_point(self, step_count):
        return step_count < self.net_config['train_step_milestone'] or  \
               step_count % self.net_config['train_step_milestone']==0
    
    def check_convergence(self, i, kpi_percent_momentum):
        is_last_iteration = (i == self.net_config['train_max_steps'])
        is_amplified_oscillation = kpi_percent_momentum > 2
        is_kpi_converged = kpi_percent_momentum < self.net_config['train_convergence_threshold']
        if is_last_iteration or is_kpi_converged: # or is_amplified_oscillation
            print("CONVERGED_EXIT_AT_ITERATION=", i)
            self.plot_step_objects(step_count=i) # last one
            self.save_objects()
            self.save_models()
            return
        
    """
    save artifact methods
    """
    def save_models(self):
        wandb.save(self.net_config['model_full_path'] + '/' + '*.h5')
           
    def save_objects(self):
        wandb.save(self.net_config['creation_full_path'] + '/*')

