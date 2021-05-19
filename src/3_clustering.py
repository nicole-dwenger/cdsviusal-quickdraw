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
  - -k, --k_clusters, int, optional, default: 5, number of clusters to generate

Output saved in ../out/3_clustering/:
  - {word}_{k_clusters}_clusters.png
"""

# LIBRARIES ---------------------------------------------------

# Basics
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.quickdraw_utils import npy_to_df

# Kmeans and VGG16
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# HELPER FUNCTIONS --------------------------------------------

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

def save_cluster_drawings(df, k_clusters, output_path):
    """
    Generate one image containing 20 examples of each cluster.
    Input:
      - df: dataframe, containg column "img_256", storing images in arrays 
            and column "cluster_labels", storing corresponding cluster labels
      - k_clusters: number of k clusters which were extracted
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
        
    # Initialise figure, size adjusted to k_clusters
    fig = plt.figure(figsize = (25, k_clusters * 2))
    # Make figure background white
    fig.patch.set_facecolor('white') 
    
    # Define rows and columns of subplots
    rows, columns = k_clusters, 20
    
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


# MAIN FUNCTION -----------------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Argument option for word
    ap.add_argument("-w", "--word", type = str, 
                    help = "Word to generate clusters for, corresponding .npy file should be in out/0_preprocessed_data",
                    required = True)
    
    # Argument option for k clusters
    ap.add_argument("-k", "--k_clusters", type = int, 
                    help = "Number of k clusters to extract",
                    required = False, default = 5)
    
    # Extract arguments
    args = vars(ap.parse_args())
    word = args["word"]
    k_clusters = args["k_clusters"]
        
    # --- DATA PREPERATION ---
    
    # Print message
    print(f"\n[INFO] Initialising VGG16 feature extraction and kmeans clustering of {k_clusters} for {word}.")

    # Get path to the file corresponding to the word and load file to df
    filepath = os.path.join("..", "out", "0_preprocessed_data", f"{word}.npy")
    df = npy_to_df([filepath], columns = ["word", "country", "img_256", "img_32"])
    
    # --- FEATURE EXTRACTION AND CLUSTERING ---
    
    # Load VGG16, without top layers and input size corresponding to images
    model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(32, 32, 3))

    # Extract features for each of the images in the dataframe
    feature_list = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        features = extract_features(row["img_32"], model)
        feature_list.append(features)
        
    # Initialise KMeans algorithm with k clusters
    kmeans = KMeans(n_clusters = k_clusters, random_state=22)
    # Fit kmeans to feature list
    kmeans.fit(feature_list)
    # Append predicted labels to dataframe
    df["cluster_labels"] = kmeans.labels_
    
    # --- OUTPUT ---
    
    # Prepare output directory 
    output_directory = os.path.join("..", "out", "3_clustering")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Define output path for image
    output_path = os.path.join("..", "out", "3_clustering", f"{word}_{k_clusters}_clusters.png")
    # Generate and save image of 20 examples for each cluster
    save_cluster_drawings(df, k_clusters, output_path)
        
    # Print message
    print(f"[INFO] Done! Image of 20 drawings for {k_clusters} clusters saved in {output_directory} for {word}!\n") 
        
if __name__=="__main__":
    main()