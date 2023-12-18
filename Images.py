## importing modules
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import IPython.display as display
from matplotlib import pyplot as plt
import cv2

import skimage.measure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imghdr

import matplotlib.image
from skimage import img_as_ubyte
from PIL import Image
import random
import hashlib
from scipy.ndimage import map_coordinates

from math import cos,sin, pi
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns

# function to get label name of numerical multiclass label 
def get_label(lab, problem = "multi"):
    if problem == "binary":
        return get_binary_label(lab)
    if lab == 0:
        return "negative"
    elif lab == 1:
        return "benign"
    else:
        return "malignant"

# function to get label name of numerical binary label 
def get_binary_label(lab):
    if lab == 0:
        return "negative"
    else:
        return "positive"

# Function to plot an image given an image and label
def plot_image(img, lab, coord = None, plot_ROI=True, grid_=None, problem = "multiclass"):
    plt.grid(grid_)
    plt.imshow(img)
    # get the target label name
    lab = get_label(lab, problem)
    plt.title(str(lab))
    # check if coordinates are available, plot_ROI is True and there are no nans
    if not coord == None and plot_ROI and not np.isnan(coord).any():
        try:
            # draw a circle using the extracted x,y,r annotations and add it to the plot
            try:
                x, y, r = coord[0]
            except:
                x, y, r = coord
            print(x, y, r)
            circle = plt.Circle((x, img.shape[0]-y), r, fill=False)
            ax = plt.gca()
            ax.add_artist(circle)
        except:
            pass
    plt.show()
    
##### Preprocessing techniques 
## function to remove noise artifacts from the image 
def noise_removal(img):
    hh, ww = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply otsu thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw largest contour as white filled on black background as mask
    mask = np.zeros((hh,ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

## function for image enhancement using CLAHE
def image_enhancement(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    backtorgb = cv2.cvtColor(equalized,cv2.COLOR_GRAY2RGB)
    return backtorgb

## function to split data into training, validation and test set 
def splitting(images, labels, coordinates, train=0.7, test=0.15, valid=0.15):
        X_train, X_test, y_train, y_test, train_coords, test_coords = train_test_split(images, labels, coordinates, test_size=test+valid, stratify=labels)
        X_valid, X_test, y_valid, y_test, valid_coords, test_coords = train_test_split(X_test, y_test, test_coords, train_size=valid/(test+valid), stratify=y_test)
        return X_train, X_valid, X_test, y_train, y_valid, y_test, train_coords, valid_coords, test_coords

## function to crop ROI based on coord (x,y,r) 
def crop_ROI(img, coord):
    x,y,r = coord
    left_bottom = (x-r,img.shape[0]-y-r)
    right_top = (x+r,img.shape[0]-y+r)
    roi = img[int(left_bottom[1]):math.ceil(right_top[1]),int(left_bottom[0]):math.ceil(right_top[0])]
    # resize the roi back to 224 x 224 
    roi = cv2.resize(roi, (224,224))
    return roi

## function to crop ROI based on a central breast region
def center_crop(image):
    # get the height and width of the input image
    h, w = image.shape[:2]
    # calculate the size of the crop (one quarter of the original image height)
    size = h // 4
    # calculate the x/y coordinates of the top-left corner of the crop
    y = (h - size) // 2
    x = (w - size) // 2
    
    # Check if center of image is in black area
    center_pixel = image[h//2, w//2,0]
    if center_pixel == 0:
        # Shift crop center until non-black pixel is found
        for i in range(1, size):
            if image[h//2+i, w//2+i] != 0:
                y += i
                x += i
                break
            elif image[h//2+i, w//2-i] != 0:
                y += i
                x -= i
                break
            elif image[h//2-i, w//2+i] != 0:
                y -= i
                x += i
                break
            elif image[h//2-i, w//2-i] != 0:
                y -= i
                x -= i
                break
    # crop the image using the calculated coordinates and size
    crop = image[y:y+size, x:x+size]
    # resize the image to 224 x 224 
    crop = cv2.resize(crop, (224, 224))
    return crop

# function to randomly crop an area of the image 
def random_crop(image):
    # get height of input image
    h = image.shape[0]
    # calculate size of crop as one quarter of input image's height
    size = h // 4
    # randomly select x/y coordinates for the crop, ensuring that the crop does not fall outside the image bounds
    y = np.random.randint(0, h - size)
    x = np.random.randint(50, h - size)
    # Extract the crop from the image using the x and y positions and the determined size
    crop = image[y:y+size, x:x+size]
    # resize crop back to 224 x 224 
    crop = cv2.resize(crop,(224,224))
    return crop

## function to rotate an image based on a specified degree theta
def rotate_image(image, lab, theta):
        try:
            # Convert the input image to a PIL Image object
            img = Image.fromarray(image)
        except: 
            # If the input image is not in the correct format, convert it and try again
            img = Image.fromarray((image*255).astype(np.uint8))
         # Rotate the image by the given angle and convert it back to a numpy array
        img_rotated = np.array(img.rotate(theta, expand = True))
        return img_rotated, lab
    
## function to rotate all images based on a list of specified degrees theta
def rotate_all(images, labels, theta = [90,180,270]):
    # define a list for the images and labels
    rotated_images = []
    rotated_labels = []
    # loop through all imags
    for img, lab in zip(images, labels):
        # loop through all degrees
        for t in theta:
            # for each image and degree, rotate the image and append it 
            rotated, lab_r = rotate_image(img,lab,t)
            rotated_images.append(rotated)
            rotated_labels.append(lab_r)
    # return an array of images and labels 
    return np.array(rotated_images), rotated_labels


## function to flip an image vertially 
def flip_vertically(image):
    # Convert numpy array to PIL image
    img = Image.fromarray(image)
    # Flip the image vertically and convert back to numpy array
    img_vertically = np.array(img.transpose(Image.FLIP_TOP_BOTTOM))
    return img_vertically
  
        
def flip_horizontally(image):
    # Convert numpy array to PIL image
    img = Image.fromarray(image)
    # Flip the image horizontally and convert back to numpy array
    img_horizontally = np.array(img.transpose(Image.FLIP_LEFT_RIGHT))
    return img_horizontally


## function to detect duplicates
def duplicate_check(images):
    def compute_hash(image):
        # Convert the image to a PIL image
        pil_image = Image.fromarray(image)
        # Convert the PIL image to a hash value
        hash_value = hashlib.md5(pil_image.tobytes()).hexdigest()
        return hash_value
    # define lists to store the indeces of duplicates and unique images
    unique_images = []
    duplicate_indeces = []
    # Create a hash set to store the hash values of the unique images in the chunk
    hash_set = set()
    # Loop through the images in the chunk and check for duplicates
    for i, image in enumerate(images):
        if i % 5000 == 0: 
            # print progress
            print(f"at index {i}")
        # Compute the hash value for the image
        hash_value = compute_hash(image)
        # If the hash value has been seen before, print a message and return the index of the duplicate image
        if hash_value in hash_set:
            print(f"Duplicate image found at index {i}")
            duplicate_indeces.append(i)
        # Otherwise, add the hash value to the hash set
        else:
            hash_set.add(hash_value)
            unique_images.append((image, hash_value))
    print(f"We are done checking. There are {len(duplicate_indeces)} duplicates")
    # return the indeces of the duplicates and unique images
    return duplicate_indeces, unique_images
    
    
## function to plot frequencies 

def plotfrequencies(labels, per = False):
    # Get the unique labels and their counts
    unique, counts = np.unique(labels, return_counts = True)
    
    # If per is true, convert counts to percentages
    if per == True:
        counts = (counts / len(labels)) * 100
        lab = "Percentages"
    else:
        lab = "Number"
        
    # Define the labels and values for the x-axis of the plot
    x_labels = ['Normal', 'Benign', 'Malignant']
    x_values = [0, 1, 2]

    # Create the bar plot
    plt.bar(x_values, counts, tick_label=x_labels)
    
    # Set the y and x axis labels and title of the plot
    plt.ylabel(f"{lab} of mammograms", fontsize = 12)
    plt.xlabel("Mammogram label", fontsize = 12)
    plt.xticks(x_values, x_labels)
    plt.title(f"Barplot of mammograms", fontsize = 12)
    
    # Add the counts/percentages above each bar in the plot
    for i, v in enumerate(counts):
        if lab == "Number":
            plt.text(i, v + 1, str(v), color = "black", fontweight = 'bold',ha='center', va='bottom')
        else:
            plt.text(i, v + 1, str(str(round(v,3))+"%"), color = "black", fontweight = 'bold',ha='center', va='bottom')
        
    # Set the y-axis limits and make sure everything is visible
    plt.ylim(0, max(counts) * 1.1)
    plt.tight_layout()
    
    # Display the plot
    plt.show()

# Function to convert multiclass label to a binary class label (normal and abnormal)
def convert_to_binary(y):
    if y == 0:
        return 0
    elif y == 1 or y == 2:
        return 1
    
# Function balance the dataset 
def balance_dataset(X, y):
    print(type(X))
    # Find the number of samples in each class
    unique, counts = np.unique(y, return_counts=True)
    minority_class_count = np.min(counts)
    num_classes = len(unique)
    
    # Find the index of samples in the minority class
    minority_class_index = np.where(y == unique[np.argmin(counts)])[0]
    
    # Find the index of samples in the majority classes
    majority_class_indices = [np.where(y == unique[i])[0] for i in range(num_classes) if i != np.argmin(counts)]
    print(majority_class_indices)
    # Randomly select samples from the majority classes
    random_indices = []
    for indices in majority_class_indices:
        random_indices.append(np.random.choice(indices, size=minority_class_count, replace=False))
    print(random_indices)
    undersampled_indices = np.concatenate([random_indices[i] for i in range(len(random_indices))] + [minority_class_index])
    print(type(undersampled_indices))

    # Use the undersampled indices to create a balanced dataset
    try:
        X_balanced = X[undersampled_indices]
        y_balanced = y[undersampled_indices]
    except:
        X_balanced = np.array(X)[undersampled_indices]
        y_balanced = np.array(y)[undersampled_indices]
    
    # return the balanced images and labels 
    return X_balanced, y_balanced

# function change brightness and contrast of an image randomly within a specified range
def change_brightness_contrast(image, brightness_range=(-15, 15), contrast_range=(0.5, 1.5)):

    # adjust brightness
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    image = np.clip(image + brightness, 0, 255)

    # adjust contrast
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    image = np.clip(contrast * (image - 128.0) + 128.0, 0, 255)

    # convert image back to uint8
    image = np.uint8(image)
    print(brightness, contrast)

    return image


# function to plot t-SNE (t-distributed stochastic neighbor embedding)
def plot_tsne_images(images, labels):
    images = np.array(images)

    # Reshape image data
    numb = images.shape[0]
    x = images.reshape((numb, -1))
    
    # Normalize the data
    X_norm = x / 255.0
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X_norm)
    
    # Create dataframe
    cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'], data=np.column_stack((X_tsne, labels)))
    
    # Cast targets column to int
    cps_df.loc[:, 'target'] = cps_df.target.astype(int)
    
    # Map targets to actual clothes for plotting
    target_map = {0:'normal', 1: 'benign', 2: 'malignant'}
    cps_df.loc[:, 'target'] = cps_df.target.map(target_map)
    
    # Plot t-SNE results
    grid = sns.FacetGrid(cps_df, hue="target", size=8)
    grid.map(plt.scatter, 'CP1', 'CP2').add_legend()
    plt.title("tsne plot")
    
    # Show plot
    plt.show()

