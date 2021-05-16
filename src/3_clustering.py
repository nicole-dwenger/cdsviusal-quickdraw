#!/usr/bin/env python

""""
Generating k clusters of images beloning to one word, using KMeans algorithm of features extracted from VGG16.  

For a preprocessed dataframe of a word (containing 6000 drawings)
  - For 32x32 imgs in dataframe: load images, and preprocesses to fit to VGG16
  - Prepare model: load VGG16 without top layers, for input size of 32x32x3
  - Extract features: predict features of each image using VGG16, returns 512 long feacture vector for each image
  - Generate k clusters using extracted features, and KMeans algorithm
  - Save an image of 20 example drawings for each cluster

Input:
  - -w, --word, str, required, examples: "yoga", "The_Mona_Lisa" 
    (should correspond to filename in out/0_preprocessed_data/)
  - -n, --n_clusters, int, optional, default: 5, number of clusters to generate

Output saved in ../out/3_clustering/:
  - {word}_{n_clusters}_clusters.png
"""

# LIBRARIES ---------------------------------------------------

# Basics
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.quickdraw_utils import npy_to_df, extract_features, save_cluster_drawings

# Kmeans and VGG16
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16


# MAIN FUNCTION -----------------------------------------------

def main():
    
    # --- ARGUMENT PARSER AND OUTPUT DIRECTORY ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Argument option for word
    ap.add_argument("-w", "--word", type = str, 
                    help = "Word to generate clusters for, corresponding .npy file should be in out/0_preprocessed_data",
                    required = True)
    
    # Argument option for k clusters
    ap.add_argument("-n", "--n_clusters", type = int, 
                    help = "Number of clusters to extract",
                    required = False, default = 5)
    
    # Extract arguments
    args = vars(ap.parse_args())
    word = args["word"]
    n_clusters = args["n_clusters"]
    
    # Prepare output directory 
    output_directory = os.path.join("..", "out", "3_clustering")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # --- DATA AND MODEL PREPARATION ---
    
    # Print message
    print(f"\n[INFO] Initialising VGG16 feature extraction and kmeans clustering of {n_clusters} for {word}.")

    # Get path to the file corresponding to the word and load file to cd
    filepath = os.path.join("..", "out", "0_preprocessed_data", f"{word}.npy")
    df = npy_to_df([filepath], columns = ["word", "country", "img_256", "img_32"])
    
    # Load VGG16, without top layers and input size corresponding to images
    model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(32, 32, 3))
    
    # --- FEATURE EXTRACTION AND CLUSTERING ---

    # Extract features for each of the images in the dataframe
    feature_list = []
    for index, row in df.iterrows():
        features = extract_features(row["img_32"], model)
        feature_list.append(features)
        
    # Initialise KMeans algorithm with k clusters, and fit to feature list
    kmeans = KMeans(n_clusters = n_clusters, random_state=22)
    kmeans.fit(feature_list)
    # Append predicted labels to dataframe
    df["cluster_labels"] = kmeans.labels_
    
    # --- OUTPUT ---
    
    # Define output path for image
    output_path = os.path.join("..", "out", "3_clustering", f"{word}_{n_clusters}_clusters.png")
    # Generate and save image of 20 examples for each cluster
    save_cluster_drawings(df, n_clusters, output_path)
        
    # Print message
    print(f"[INFO] Done! Image of 20 drawings for {n_clusters} clusters saved in {output_directory} for {word}!\n") 
        
if __name__=="__main__":
    main()