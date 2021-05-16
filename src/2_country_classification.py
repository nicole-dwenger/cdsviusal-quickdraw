#!/usr/bin/env python

""""
Classification of countries (DE, RU, US) of preprocessed QuickDraw sketches beloning to a 
single word using transfer learning with VGG16. 

For a preprocessed dataframe of a word (containing 2000 drawings for each country: DE, US, RU)
  - Turn 32x32 imgs to array and normalise = X
  - Turn list of corresponding "country" labels to array, binarize = y
  - Split X and y into train/test, with 80/20 splot
  - Prepare VGG16: remove fully connected layers at the top, append flattening, dense and output layer
  - Train appended layers to classify countries of sketches, with input batch size and epochs
  - Evaluate model by predicting country labels of test data
  - Save model information, model history and classification report

Input:
  - -w, --word, str, required, word to classify drawings for, 
     corresponding .npy file should be in out/0_preprocessed_data/"
  - -b, --batch_size, int, optional, default: 40, size of batch to train model on within an epoch
  - -e, --epoch, int, optional, defualt: 20, number of epochs to train model for 

Output saved in ../out/2_country_classification/{word}:
  - model_summary.txt: summary of model architecture
  - model_plot.png: plot of model architecture
  - model_history.png: plot of model training history
  - model_report.txt: classification report of model
"""

# LIBRARIES ---------------------------------------------------

# Basics
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Utilities
import sys
sys.path.append(os.path.join(".."))
from utils.quickdraw_utils import (npy_to_df, prepare_data, 
                                   save_model_info, save_model_history, save_model_report)

# ML tools
from sklearn.metrics import classification_report

# Tensorflow keras, VGG16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (Flatten, Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# MAIN FUNCTION -----------------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Argument option for word
    ap.add_argument("-w", "--word", type = str, 
                    help = "Word to classify drawings for, corresponding .npy file should be in out/0_preprocessed_data",
                    required = True)
    
    # Argument option for batch size
    ap.add_argument("-b", "--batch_size", type = int, 
                    help = "Size of batch of images to train the model",
                    required = False, default = 40)
    
    # Argument option for the number of epochs
    ap.add_argument("-e", "--epochs", type = int, 
                     help = "Number of epochs to train the model",
                     required = False, default = 20)
    
    # Extract arguments of batch size and epochs
    args = vars(ap.parse_args())
    word = args["word"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    
    # --- DATA PREPARATION ---
    
    print("\n[INFO] Initialising classification of countries of sketches using the pretrained model VGG16!")

    # Load corresponding .npy file of the word and turn into df
    filepath = [os.path.join("..", "out", "0_preprocessed_data", f"{word}.npy")]
    df = npy_to_df(filepath, columns = ['word','country',"img_256", "img_32"])
    
    # Prepare images/drawings [X] and corresponding labels/words [x], and save unique label names (countries)
    X_train, X_test, y_train, y_test, label_names = prepare_data(df, "img_32", "country")
    # Retrieve number of labels (countries), for output layer in model
    n_labels = len(label_names)
    
    # --- MODEL PREPARATION ---
    
    print("[INFO] Preparing VGG16 and adjusting fully connected layers!")
    
    # Load VGG16 without fully connected layers and for image size of 32x32x3
    model = VGG16(include_top=False, pooling='avg', input_shape=(32, 32, 3))

    # Set layers in the model to not be trainable
    for layer in model.layers:
        layer.trainable = False
    
    # Define fully connected layers to fit to the data
    flat_layer = Flatten()(model.layers[-1].output)
    dense_layer = Dense(256, activation='relu')(flat_layer)
    dropout_layer = Dropout(0.2)(dense_layer)
    output_layer = Dense(n_labels, activation='softmax')(dropout_layer)

    # Append output layers to the model
    model = Model(inputs=model.inputs, 
                  outputs=output_layer)
        
    # Compile model 
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # --- MODEL TRAINING AND EVALUATION ---
    
    print("[INFO] Training and evaluating model!")
    
    # Train the model to classify countries
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1)
    
    # Evaluate model, by generating predictions for test data
    predictions = model.predict(X_test, batch_size=batch_size)
    # Generate classification report with correct label names
    report = classification_report(y_test.argmax(axis=1), 
                                   predictions.argmax(axis=1), 
                                   target_names=label_names)
    
    # Prepare output directory
    output_directory = os.path.join("..", "out", "2_country_classification", word)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Save model info, history and classification report 
    save_model_info(model, output_directory, "model_summary.txt", "model_plot.png")
    save_model_history(history, epochs, output_directory, "model_history.png")
    save_model_report(report, epochs, batch_size, output_directory, "model_report.txt")
    
    # Print classification report
    print(f"[OUTPUT] Classification report for {word} model trained with batch size {batch_size} and {epochs} epochs:\n {report}")
    
    # Print message
    print(f"\n[INFO] All done! Output is saved in {output_directory}")

    
if __name__=="__main__":
    main()