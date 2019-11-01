#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Config
import onenow_config as oconfig


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[3]:


import glob
import os
from os import listdir
from os.path import isfile, join

file_names = [f for f in listdir(oconfig.image_full_path) if isfile(join(oconfig.image_full_path, f))]
print(file_names)

qualified_files = glob.glob(oconfig.image_full_path + '/' + '*') # * means all if need specific format then *.csv
sorted_files = sorted(qualified_files, key=os.path.getctime, reverse=True)
# print(sorted_files)


# In[4]:


import onenow_utils as outil


# In[5]:


sample_images = []

i=0
for file_path in sorted_files: 
    sample_images.append(file_path)
    i += 1


# In[6]:


image_width_height = 12
grid_cells_xy = 4

plt.figure(figsize=(image_width_height, image_width_height))

i=0
for image in sample_images:
    plt.subplot(grid_cells_xy, grid_cells_xy, i+1)
    img = mpimg.imread(image)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    i += 1

plt.tight_layout()
plt.show()
print("best_image=", sample_images[0])


# In[ ]:





# In[ ]:





# In[ ]:




