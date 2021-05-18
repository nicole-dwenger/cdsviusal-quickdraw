#!/usr/bin/env python

"""
Utility functions for quickdraw project:

For data preprocessing: 
  - npy_to_df: load .npy file(s) with preprocessed drawings and turn into df
  - prepare_data: preprocess images and labels, turn into arrays
    normalise images and binarize labels and split test train

For model output: 
  - unique_path: enumerate filename, if it exists already
  - save_model_info: save model summary as txt and visualistion of model architecture as png
  - save_model_history: save plot of model history as png
  - save_model_report: save classification report as txt
"""

# LIBRARIES ---------------------------------------------------

# Basics
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# ML
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Tensorflow, VGG16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warnings
from tensorflow.keras.utils import plot_model


# DATA PREPROCESSING ---------------------------------------------------

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

     
if __name__=="__main__":
    pass