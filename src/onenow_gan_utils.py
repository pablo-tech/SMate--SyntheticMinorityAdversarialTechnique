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
    ## first delete it
    # try:
    #    shutil.rmtree(full_path) 
    # except Exception:
    #    print("DELETION_FAILED=%s" % full_path)
        
    ## now create it
    try:
        os.mkdir(full_path)
    except OSError:
        print ("CREATION_FAILED=%s failed" % full_path)
        return
    print("CREATION_SUCCEEDED=%s" % full_path)


def plot_sub_objects(sample_images, file_name, full_path):
    """
    Plots image containing subplots 
    """
    try:
        figure_width = 12
        figure_height = 12
        plt.figure(figsize=(figure_width, figure_height))

        sample_size = sample_images.shape[0]
        num_items_subplot_x = 5
        num_items_subplot_y = 5

        for i in range(sample_size):
            subplot_index = i+1
            plt.subplot(num_items_subplot_x, num_items_subplot_y, subplot_index)
            image = sample_images[i, :, :, :]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.savefig(full_path + '/' + file_name)
        plt.close('all')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(e)    
        
def plot_one_object(image, file_name, full_path):
    """
    Plots image containing subplots 
    """
    try:
        # plot
        figure_width = 12
        figure_height = 12    
        plt.figure(figsize=(figure_width, figure_height))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(full_path + '/' + file_name)
        plt.close('all')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(e)    
    
def get_reshaped_image(sample_object, object_rows, object_cols, object_channels):
    """
    Numpy reshape image per desired rows, columns, and channels
    """
    try:
        reshaped = np.reshape(sample_object, [object_rows, object_cols, object_channels])
    except Exception as e:
        print(e)    
    return reshaped
    

def setup_file_logger(logger, log_file):
    """
    Initializes formatted logger 
    """
    try:
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s') # %(levelname)s 
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.INFO)
    except Exception as e:
        print(e)      
    
    