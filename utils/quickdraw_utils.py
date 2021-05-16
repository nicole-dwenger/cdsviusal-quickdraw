#!/usr/bin/env python

"""
Utility functions for quickdraw project.

For data preprocessing: 
  - strokes_to_img: turn strokes of quickdraw sketches into an image
  - npy_to_df: load .npy file(s) with preprocessed drawings and turn into df
  - prepare_data: preprocess images and labels, turn into arrays
    normalise images and binarize labels and split test train

For model output: 
  - unique_path: enumerate filename, if it exists already
  - save_model_info: save model summary as txt and visualistion of model architecture as png
  - save_model_history: save plot of model history as png
  - save_model_report: save classification report as txt

For feature extraction and clustering: 
  - extract_features: extracts features of images using a model
  - save_cluster_drawings: get drawings and labels from dataframe,
    and create iamge of 20 examples for each cluster
"""

# LIBRARIES ---------------------------------------------------

# Basics
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import cv2

# ML
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Tensorflow, VGG16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warnings
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# DATA PREPROCESSING ---------------------------------------------------

def strokes_to_img(strokes, output_size):
    """
    Turn drawing strokes into image with strokes coloured by time of occuance.
    Input: 
      - strokes: List of strokes of quickdraw images
      - ouput_size: output size of image    
    Returns:
      - img: with 3 colour channels, and sized to output size
    """
    # Create empty array of original image size (0 = black)
    img = np.zeros((256, 256, 1), np.uint8)
    
    # For each stroke in the drawing
    for stroke in strokes:
        # Get all x points and all y points
        stroke_x = (np.array(stroke[0]))
        stroke_y = (np.array(stroke[1]))        
        
        # For each of the x and y pairs
        for i in range(len(stroke_x) - 1):
            # Get x and y coordinates for a point and the following point
            p1 = (stroke_x[i], stroke_y[i])
            p2 = (stroke_x[i+1], stroke_y[i+1])
            # Draw line between the points (255 = white) 
            img = cv2.line(img, p1, p2, (255), 2)
    
    # Invert: background white, drawing black
    img_grey = cv2.bitwise_not(img)
    # Convert to have 3 color channels
    img_rgb = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2RGB)
    # Resize to 64 for CNN
    img_sized = cv2.resize(img_rgb, (output_size, output_size), interpolation = cv2.INTER_AREA)
            
    return img_sized

def npy_to_df(filepaths, columns):
    """
    Read one or more .npy file(s) and turn into one dataframe
    Input:
      - filepaths: List of filepath(s)
      - columns: column names of output dataframe
    Returns:
      - dataframe, with column names and all .npy files concatenated
    """
    # Create target dataframe
    df = pd.DataFrame(columns = columns)
    
    # For each filepath
    for path in filepaths: 
        # Load npy file as array
        npy = np.load(path, allow_pickle = True)
        # Turn npy to dataframe with column names
        df_word = pd.DataFrame(npy, columns = columns)
        # Concatenate to dataframe
        df = pd.concat([df, df_word])
    
    return df
    
def prepare_data(df, img_type, label_type):
    """
    Preprocess images and labels as model input
    Input:
      - df: processed dataframe with images and relevant labels
      - img_type: "img_32" or "img_224"
      - label_type: "word" for word classifier or "country" for country classifier
    Returns:
      - X_train, X_test, y_train, y_test, unique, sorted label_names
    """
    # From df get images and labels and turn into array
    X_array = np.array(df[img_type].tolist())
    y_array = np.array(df[label_type].tolist())

    # Normalise images
    X_scaled = (X_array - X_array.min())/(X_array.max() - X_array.min())

    # Binarize labels
    lb = LabelBinarizer()
    y_binary = lb.fit_transform(y_array)
    # Get label names 
    label_names = sorted(np.unique(y_array))
    
    # Split x and y
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, random_state=9, test_size=0.2)
    
    return X_train, X_test, y_train, y_test, label_names


# MODEL OUTPUT ---------------------------------------------------

def unique_path(filepath):
    """
    Create unique filename by enumerating if path exists already 
    Input:
      - desired fielpath
    Returns:
      - filpath, enumerated if it exists already
    """ 
    # If the path does not exist
    if not os.path.exists(filepath):
        # Keep the original filepath
        return filepath
    
    # If path exists:
    else:
        i = 1
        # Split the path and append a number
        path, ext = os.path.splitext(filepath)
        # Add extension
        new_path = "{}_{}{}".format(path, i, ext)
        
        # If the extension exists, enumerate one more
        while os.path.exists(new_path):
            i += 1
            new_path = "{}_{}{}".format(path, i, ext)
            
        return new_path
    
def save_model_info(model, output_directory, filename_summary, filename_plot):
    """
    Save model summary in .txt file and plot of model in .png
    Input:
      - model: compiled model
      - output_directory: path to output directory
      - filename_summary: name of file to save summary in
      - filename_plot: name of file to save visualisation of model
    """
    # Define path fand filename for model summary
    out_summary = unique_path(os.path.join(output_directory, filename_summary))
    # Save model summary in defined file
    with open(out_summary, "w") as file:
        with redirect_stdout(file):
            model.summary()

    # Define path and filename for model plot
    out_plot = unique_path(os.path.join(output_directory, filename_plot))
    # Save model plot in defined file
    plot_model(model, to_file = out_plot, show_shapes = True, show_layer_names = True)
     
def save_model_history(history, epochs, output_directory, filename):
    """
    Plotting the model history, i.e. loss/accuracy of the model during training
    Input: 
      - history: model history
      - epochs: number of epochs the model was trained on 
      - output_directory: desired output directory
      - filename: name of file to save history in
    """
    # Define output path
    out_history = unique_path(os.path.join(output_directory, filename))

    # Visualize history
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_history)

def save_model_report(report, epochs, batch_size, output_directory, filename):
    """
    Save report to output directory
    Input: 
      - report: model classifcation report
      - output_directory: final output_directory
      - filename: name of file to save report in
    """
    # Define output path and file for report
    report_out = unique_path(os.path.join(output_directory, filename))
    # Save report in defined path
    with open(report_out, 'w', encoding='utf-8') as file:
        file.writelines(f"Classification report for model trained with {epochs} epochs and batchsize of {batch_size}:\n")
        file.writelines(report) 

        
# FEATURE EXTRACTION AND CLUSTERING ---------------------------------------------------       
        
def extract_features(image, model):
    """
    Extract features for a given image, using pretrained layers of a given model
    Input:
      - img: image, size should be as input shape of model
      - model: loaded, pretrained model, here: vgg16
    Returns: 
      - flattened, normalised features of the image
    """
    # Turn image into array
    img = np.array(image)
    # Expand dimension (1 at the beginning)
    expanded_img = np.expand_dims(img, axis=0)
    # Preprocess image, to fit with vgg16
    preprocessed_img = preprocess_input(expanded_img)
    # Predict the features using the pretrained model
    features = model.predict(preprocessed_img)
    # Flatten the features
    flattened_features = features.flatten()
    # Normalise features 
    normalized_features = flattened_features / np.linalg.norm(features)
    
    return normalized_features


def save_cluster_drawings(df, n_clusters, output_path):
    """
    Generate one image containing 20 examples of each cluster.
    Input:
      - df: dataframe, containg column "img_256", storing images in arrays 
            and column "cluster_labels", storing corresponding cluster labels
      - n_clusters: number of clusters which were extracted
      - output_path: output path, where image should be stored 
    """
    # Create empty lists for images of clusters
    cluster_images = []
    # Create empty list for cluster labels of images
    cluster_labels = []

    # For each cluster of the possible cluster labels:
    for cluster in set(df["cluster_labels"].tolist()):
        # Filter only images belonding to the cluster, get the first 20
        cluster_sample = df[df["cluster_labels"] == cluster][:20]
        # Append images to the cluster_images list, 256 for better quality
        cluster_images.extend(cluster_sample["img_256"].tolist())
        # Append cluser labels of images to the cluster labels list
        cluster_labels.extend(cluster_sample["cluster_labels"].tolist())
        
    # Initialise figure, size adjusted to n_clusters
    fig = plt.figure(figsize = (25, n_clusters * 2))
    # Make figure background white
    fig.patch.set_facecolor('white') 
    
    # Define rows and columns of subplots
    rows, columns = n_clusters, 20
    
    # For each image in cluster_images, plot it as a subplot
    # And append label as title
    for i in range(1, len(cluster_images) + 1):
        img = cluster_images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Cluster {cluster_labels[i-1]}")
        
    # Save figure in output_path
    plt.savefig(output_path)
     
if __name__=="__main__":
    pass