#!/usr/bin/env python

"""
Classification of words of preprocessed QuickDraw sketches using transfer learning with VGG16. 

For all preprocessed files in out/0_preprocessed_data/
  - Preprocess X (images in 32x32x3), and y (labels, words): normalise images, binarize labels
  - Split X and y into train/test, with 80/20 split
  - Prepare VGG16: remove fully connected layers at the top, append flattening, dense and output layer
  - Train appended layers to classify words of sketches, with input batch size and epochs
  - Evaluate model by predicting labels of test data
  - Save model information, model history and classification report

Input:
  - -b, --batch_size, int, optional, default: 40, size of batches to train model on
  - -e, --epoch, int, optional, defualt: 5, number of epochs to train model for 

Output saved in ../out/1_word_classification/:
  - model_summary.txt: summary of model architecture
  - model_plot.png: plot of model architecture
  - model_history.png: plot of model training history
  - model_report.txt: classification report of model
"""


# LIBRARIES ---------------------------------------------------

# Basics
import os
import glob
import argparse
import numpy as np
import pandas as pd

# Utility function
import sys
sys.path.append(os.path.join(".."))
from utils.quickdraw_utils import (npy_to_df, prepare_data, 
                                   save_model_info, save_model_history, save_model_report)

# ML tools
from sklearn.metrics import classification_report

# Tensorflow Keras, VGG16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warnings
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# MAIN FUNCTION -----------------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Argument option for batch size
    ap.add_argument("-b", "--batch_size", type = int, 
                    help = "Size of batch of images to train the model",
                    required = False, default = 40)
    
    # Argument option for the number of epochs
    ap.add_argument("-e", "--epochs", type = int, 
                     help = "Number of epochs to train the model",
                     required = False, default = 5)
    
    # Extract arguments of batch size and epochs
    args = vars(ap.parse_args())
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    
    # --- DATA PREPARATION ---
    
    print("\n[INFO] Initialising classification of words of sketches using the pretrained model VGG16.")
    
    # Get filepaths of all preprocessed .npy files
    filepaths = glob.glob(os.path.join("..", "out", "0_preprocessed_data", "*.npy"))
    # Create target dataframe for data, based on preprocessed .npy files
    df = npy_to_df(filepaths, columns = ["word","country","img_256", "img_32"])
        
    # Preprocess images/drawings [X] and corresponding labels/words [x], and save unique label names (words)
    # This scales images, binarizes labels and also returns the sorted, unique label names
    X_train, X_test, y_train, y_test, label_names = prepare_data(df, "img_32", "word")
    # Retrieve number of labels (words), for output layer in model
    n_labels = len(label_names)
    
    # --- MODEL PREPARATION ---
    
    print("[INFO] Preparing VGG16 and adjusting fully connected layers.")
    
    # Load VGG16 without fully connected layers and for image size of 32x32x3
    model = VGG16(include_top=False, pooling='avg', input_shape=(32, 32, 3))

    # Set layers in the model to not be trainable
    for layer in model.layers:
        layer.trainable = False
    
    # Define fully connected layers to fit to the data
    flat_layer = Flatten()(model.layers[-1].output)
    dense_layer = Dense(256, activation="relu")(flat_layer)
    output_layer = Dense(n_labels, activation="softmax")(dense_layer)

    # Append output layers to model
    model = Model(inputs=model.inputs, 
                  outputs=output_layer)
        
    # Compile model 
    model.compile(optimizer=Adam(lr=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    # --- MODEL TRAINING AND EVALUATION ---
    
    print(f"[INFO] Training and evaluating model with batch size: {batch_size}, epochs: {epochs}.")
    
    # Train the model to classify words
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1)
    
    # Evaluate the model by generating predictions for test data
    predictions = model.predict(X_test, batch_size=batch_size)
    # Get classification report with correct label names
    report = classification_report(y_test.argmax(axis=1), 
                                   predictions.argmax(axis=1), 
                                   target_names=label_names)
    
    # Prepare output directory
    output_directory = os.path.join("..", "out", "1_word_classification")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Save model info, history and classification report 
    save_model_info(model, output_directory, "model_summary.txt", "model_plot.png")
    save_model_history(history, epochs, output_directory, "model_history.png")
    save_model_report(report, epochs, batch_size, output_directory, "model_report.txt")
    
    # Print classification report
    print(f"[OUTPUT] Classification report for model:\n")
    print(report)
    
    # Print message
    print(f"\n[INFO] All done! Output is saved in {output_directory}.")

   
if __name__=="__main__":
    main()