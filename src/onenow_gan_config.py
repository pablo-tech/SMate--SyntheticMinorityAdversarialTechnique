import onenow_gan_utils as outil


class SystemConfig(object):
    """
    Overall system configuration
    """        
    def __init__(self, project_name): 
        self.sys_config = self.get_configuration(project_name)
        

    def get_configuration(self, project_name):

        sys_config = {}

        ## general
        sys_config['project_name'] = project_name
        
        ## path
        sys_config['root_folder'] = '/home/ec2-user/SageMaker/efs/' 
        # sys_config['root_folder'] = '/home/ec2-user/SageMaker/Rosenblatt-AI/CS230/' 
        sys_config['log_root'] = sys_config['root_folder'] + 'log/'
        sys_config['creation_root'] =  sys_config['root_folder'] + 'creation/'        
        sys_config['model_root'] =  sys_config['root_folder'] + 'model/'                 

        ## train
        sys_config['train_max_steps'] = 10*10**3
        sys_config['train_adaptation_threshold'] = 100
        sys_config['train_step_milestone'] = 100 # step count
        sys_config['train_convergence_threshold'] = 5*10**(-2) # key metric oscillating at %
        
        return sys_config
    
    """
    Getters and setters
    """
    def get_config(self):
        return self.sys_config
    
    def set_log_file(self, logger, param_name):
        log_file = self.sys_config['log_root'] + 'out.' + self.sys_config['project_name'] + param_name +'.log'
        outil.setup_file_logger(logger, log_file)
        return log_file
    
    def set_creation_folder(self, param_name):
        creation_full_path = self.sys_config['creation_root'] + self.sys_config['project_name'] + param_name 
        self.sys_config['creation_full_path'] = creation_full_path
        outil.create_folder(creation_full_path)
        return creation_full_path
    
    def set_model_folder(self, param_name):
        model_full_path = self.sys_config['model_root'] + self.sys_config['project_name'] + param_name
        self.sys_config['model_full_path'] = model_full_path
        self.sys_config['discriminator_model_path'] = model_full_path + '/' + "discriminator.h5"
        self.sys_config['generator_model_path'] = model_full_path + '/' + "generator.h5"
        outil.create_folder(model_full_path)
        return model_full_path