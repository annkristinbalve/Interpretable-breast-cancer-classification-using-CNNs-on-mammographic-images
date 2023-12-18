## importing modules
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import IPython.display as display
from matplotlib import pyplot as plt
import skimage.measure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2
import imghdr
import matplotlib.image
from skimage import img_as_ubyte
from PIL import Image
import random
import numpy as np
import hashlib
from PIL import Image
from scipy.ndimage import map_coordinates
from math import cos,sin, pi
from sklearn.model_selection import train_test_split

## code adapted from: https://www.kaggle.com/code/codeer/mias-mammography-cnn
## MIAS class for loading, transforming, and visualizing the images
class MIAS:
    def __init__(self, dir_images, txt_file, pix = 224): # 224
        
         # Read the text file
        df = pd.read_csv(txt_file, sep=" ").drop('Unnamed: 7',axis=1)
        # Add column names to the data frame
        df.columns = ["REFNUM","BG","CLASS","SEVERITY","X","Y","RADIUS"]
    
        # Get the list of image filenames
        image_files = os.listdir(dir_images)
        image_files = [f for f in image_files if f.endswith('.pgm')]
        
        # Initialize lists for images, labels, and coordinates
        X = []
        Y = []
        coordinates = []
        #abnormalities = []
        
       # Loop over the image files
        for i, image_file in enumerate(image_files):
            # Get the image ID from the filename
            filename = image_file.split(".")[0]
            image_id = int(image_file[3:6])
            
            # Load the image
            image_path = os.path.join(dir_images, image_file)
            image = cv2.imread(os.path.join(dir_images,image_file))
            image = cv2.resize(image, (pix,pix))
            
            # Append the image to the X list
            X.append(image)
            
            # Get the label for the image
            label = df.loc[df['REFNUM'] == filename, 'SEVERITY'].values[0]
            if label == "B":
                label = 1
            elif label == "M":
                label = 2
            else:
                label = 0
            ## Append label to Y list
            Y.append(label) 
           # abnormalities.append(df.loc[df['REFNUM'] == filename, 'CLASS'].values[0]) ## append abnormality type
            
            # Get the coordinates for the image
            coords = []
            coords_df = df.loc[df['REFNUM'] == filename, ['X', 'Y', 'RADIUS']]
            for index, row in coords_df.iterrows():
                # extract the x,y coordinates and radius and rescale all values to account for the image resizing
                x,y,r = map(lambda x: x * (pix/1024), [row['X'],row['Y'],row['RADIUS']]) 
                coords.append((x,y,r))
            # Append coordinates to coordinate list
            coordinates.append(coords) 
        # Save all images, labels and coordinates as model attributes
        self.images = X
        self.labels = Y
        self.coordinates = coordinates
      #  self.abnormalities = abnormalities 
        self.pixels = len(self.images[0])

    # Function to plot one image given provided image index
    def plot_image(self, idx, plot_ROI = True, grid_ = None):
        #fig, ax = plt.subplots()
        plt.grid(grid_)
        plt.imshow(self.images[idx])
        lab = self.labels[idx]
        lab = self.get_label(lab)
        plt.title(str(lab)+" (idx.:" + str(idx) + ")")
        # check whether coordinates are nan values and plot_ROI argument is True 
        if plot_ROI and self.coordinates[idx][0][0].astype(str)!= "nan":
        # Mark abnormality if x/y/r annotations are available
            try:  
                x,y,r = self.coordinates[idx][0]
                print(x,y,r)
                # Draw a circle based on the annotations and add it to the plot
                circle = plt.Circle((x,self.pixels - y),r,fill = False) 
                ax = plt.gca()
                ax.add_artist(circle)
            except:
                pass
        plt.show()
      
    # Function to plot frequencies with a barplot (per = percentage specifying whether counts should be displayed as percentages)
    def plotfrequencies(self, per = False):
        unique, counts = np.unique(self.labels, return_counts = True)
        if per == True:
            counts = (counts / len(self.labels)) * 100
            lab = "Percentages"
        else:
            lab = "Frequencies"
        plt.bar(unique, counts)
        plt.title(f"{lab} of mammogram images")
        plt.show()
        
    # Function to plot unique images
    def plot_unique(self):
        # find unique classes
        classes = np.unique(self.labels)
        # create dict with empty lists for each class
        class_dict = {}
        for c in classes:
            class_dict[c] = []
        # loop through labels and append corresponding images to class_dict
        for i in range(len(self.labels)):
            label = self.labels[i]
            image = self.images[i]
            class_dict[label].append([image,i])
       
        fig, ax = plt.subplots(1, len(classes), figsize=(20,5))
        # select one random image for each class and plot it 
        for i, c in enumerate(classes):
            images_list = class_dict[c]
            random_image = random.choice(images_list)
            img, idx = random_image
            ax[i].imshow(img)
            ax[i].set_title(f"{self.get_label(c)}", fontsize = 12)
            ax[i].axis("off") 
            # check whether coordinates are nan values and mark abnormality if x/y coordinates are available
            if self.coordinates[idx][0][0].astype(str)!= "nan": 
                try: 
                    x,y,r = self.coordinates[idx][0]
                    print(x,y,r)
                    # Draw a circle based on the annotations and add it to the plot
                    circle = plt.Circle((x,self.pixels - y),r,fill = False) 
                    ax[i].add_artist(circle)  
                except:
                    pass
        plt.tight_layout()
       # fig.savefig("METHODS/unique_images_plot")
        plt.show()
        
    # Function to get the name of the label 
    def get_label(self,lab):
        if lab == 0:
            return "negative"
        elif lab == 1:
            return "benign"
        else:
            return "malignant"
   
 
# Printing number of images of the dataset
    def __repr__(self):
        return f"There are {len(self.images)} images, of which {self.labels.count(0)} are normal {self.labels.count(1)} are benign and {self.labels.count(2)} are malignant"
    