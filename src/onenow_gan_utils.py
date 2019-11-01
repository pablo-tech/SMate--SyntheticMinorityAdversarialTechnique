import math
import os
import logging
from matplotlib import pyplot as plt
import numpy as np
import shutil



def create_folder(full_path):
    """
    Create folder in path
    """
    try:
        shutil.rmtree(full_path) 
    except Exception:
        print("DELETION_FAILED=%s" % full_path)
    try:
        os.mkdir(full_path)
    except OSError:
        print ("CREATION_FAILED=%s failed" % full_path)
        return
    print("CREATION_SUCCEEDED=%s" % full_path)

    
def plot_objects(object_rows, object_cols, sample_images, file_name, full_path):
    """
    Plots image containing subplots 
    """
    sample_size = sample_images.shape[0]
    plt.figure(figsize=(10,10))
    for i in range(sample_size):
        plt.subplot(4, 4, i+1)
        image = sample_images[i, :, :, :]
        image = np.reshape(image, [object_rows, object_cols])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(full_path + '/' + file_name)
    plt.close('all')
    # model.save(os.path.join(wandb.run.dir, "mymodel.h5"))
    plt.tight_layout()
    plt.show()
    

def setup_file_logger(logger, log_file):
    """
    Initializes formatted logger 
    """
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(message)s') # %(levelname)s 
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
       
    
    