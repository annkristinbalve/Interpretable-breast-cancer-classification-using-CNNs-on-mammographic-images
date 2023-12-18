import lime
import lime.lime_image
from lime import lime_image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import contextlib
import io

import tensorflow as tf
import numpy as np
import cv2
import tensorflow as tf

import matplotlib.cm as cm
from keras.preprocessing import image
from skimage import img_as_ubyte
from skimage.transform import resize
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
# set font size for consistent plots


@contextlib.contextmanager
# supress unnecessary output
def silent():
    new_target = io.StringIO()
    with contextlib.redirect_stdout(new_target):
        yield
    return new_target.getvalue()

 # code adapted from https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions
def plot_lime_explanation(idx, model, image, true_label, positive_only = True, hide_rest = False, random_state = 44):
    plt.rcParams['font.size'] = 16

    labels = ["Normal", "Benign", "Malignant"]
    # Create the explainer
    explainer = lime_image.LimeImageExplainer(random_state=random_state)  # random_state for reproducibility

    # Get the predict function of the model
    def predict_fn(x):
        with silent():
            return model.predict(x)

    start_time = time.time()

    # Compute LIME explanation
    with silent():  # suppress redundant output
        explanation = explainer.explain_instance(image, predict_fn)
    end_time = time.time()
    time_taken = end_time - start_time


    # Get the LIME explanation image and mask
    pred_class = np.argmax(model.predict(image[np.newaxis, :, :, :]))  ## get predicted class
    temp, mask = explanation.get_image_and_mask(pred_class, positive_only=positive_only,
                                                hide_rest=hide_rest)


    # Plot the original image, the mask, and the pixels that contributed and did not contribute to the prediction
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle(f"LIME Explanations (Predicted: {labels[pred_class]})")
    # plot the original image
    axs[0].imshow(image)
    axs[0].set_title(f"Original Image ({labels[true_label]}), idx: {idx}")

    # plot the mask 
    axs[1].imshow(mask)
    axs[1].set_title("Mask")
    
    # Define color maps for the mask and filled mask
    border_cmap = ListedColormap(['yellow'])
    print(list(np.unique(mask)))
    if list(np.unique(mask)) == [0, 1]:
        print("..")
        filled_cmap = ListedColormap(['green', 'white'])
    else:
        filled_cmap = ListedColormap(['red', 'green']) # red for negative predictions, green for positive


    # plot the image with mask and mask border marked in different colors 
    border = mark_boundaries(image, mask, mode='thick', outline_color=(1,1,0))
    filled_mask = np.ma.masked_where(mask == 0, mask)
    axs[2].imshow(border, cmap = border_cmap, alpha = .7)
    axs[2].imshow(filled_mask, alpha=0.1, cmap = filled_cmap)
    axs[2].set_title("Image with Mask")
    
    
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Time taken to compute LIME explanations: {time_taken:.2f} seconds")
    # returns image, mask, border, filled_mask, taken time
    return image, mask, border, filled_mask, time_taken


def get_heatmap(vectorized_image, model, last_conv_layer, pred_index=None):
    # This function generates a heatmap indicating the regions of an input image that the model
    # used to make its prediction. It takes as input the vectorized image, the trained model, 
    # the name of the last convolutional layer in the model, and an optional index for the prediction 
    # class. If the index is not provided, it uses the index of the class with the highest predicted 
    # probability.

    # Add a batch dimension to the vectorized image
    vectorized_image = np.expand_dims(vectorized_image, axis=0)

    # Create a new model that outputs the activations of the last convolutional layer and the final predictions
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    # Use a GradientTape to record the gradients of the predicted class with respect to the last conv layer activations
    with tf.GradientTape() as tape:
        # Get the last conv layer output and the predictions from the gradient model
        last_conv_layer_output, preds = gradient_model(vectorized_image)
        # If a prediction index is not provided, use the class with the highest predicted probability
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        # Get the predicted probability for the selected class
        class_channel = preds[:, pred_index]

    # Compute the gradients of the predicted class with respect to the last conv layer activations
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # Compute the mean of the gradients across the spatial dimensions of the activations
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Multiply each activation map by its corresponding gradient score and sum the results
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # Normalize the heatmap
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Return the heatmap as a numpy array
    return heatmap.numpy()



def superimpose_gradcam(img, heatmap, output_path=None, alpha=0.5):
    # Normalize heatmap values to 0-255 range and convert to jet colormap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap.astype('int')]

    # Convert heatmap to uint8 format and resize it to match original image size
    jet_heatmap = img_as_ubyte(jet_heatmap)
    jet_heatmap = resize(jet_heatmap, img.shape[:2], preserve_range=True, anti_aliasing=True)

    # Convert original image to uint8 format and blend it with heatmap using alpha parameter
    img_uint8 = img_as_ubyte(img)
    superimposed_img = cv2.addWeighted(img_uint8, 1 - alpha, jet_heatmap, alpha, 0, dtype=cv2.CV_8U)

    # Save output image to file if output_path parameter is provided
    if output_path is not None:
        cv2.imwrite(output_path, superimposed_img)

    return superimposed_img


# Function to visualize the original image, the Grad-CAM heatmap, and combined heatmap with image based on a model, image and conv_layer
def visualize_gradcam(img, model, last_conv_layer, true_label=None, pred_label=None, alpha=0.5, idx = None):
    labels = ["Normal", "Benign", "Malignant"]

    # Generate heatmap using the get_heatmap function
    time_start = time.time()
    heatmap = get_heatmap(img, model, last_conv_layer)
    time_end = time.time()
    time_taken = time_end - time_start 
    # Create a 1x3 subplot to visualize the original image, heatmap, and superimposed image
# Create a 1x3 subplot to visualize the original image, heatmap, and superimposed image
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))     
    fig.suptitle(f"Grad-CAM Explanations (Predicted: {labels[pred_label]})")

    # Plot original image with labels
    axs[0].imshow(img)
    axs[0].set_title(f"Original Image ({labels[true_label]}), idx: {idx}")

    # Plot heatmap
    axs[1].imshow(heatmap)
    axs[1].set_title("Heatmap")

    # Plot superimposed image
    try:
        superimposed_img = superimpose_gradcam(img, heatmap, alpha=alpha)
        
    except:
        # If the heatmap is uniform, the model did not identify any important regions
        print("The heatmap is uniform, indicating that the model did not identify any important regions for the predicted class.")
        superimposed_img = img
       # axs[2].set_title("Image")
    axs[2].imshow(superimposed_img)
    axs[2].set_title(f"Image with Heatmap")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print the total time taken to compute the Grad-CAM explanation
    print(f"Time to compute Grad-CAM explanations: {time_taken:.4f} seconds")
    # returns time, original image, gradcam-heatmap and the superimposed image
    try:
        return time_taken, img, heatmap, superimposed_img
    except:
        return time_taken, img, heatmap, None
