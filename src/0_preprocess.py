#!/usr/bin/env python

"""
Script to preprocess QuickDraw data
- In the raw data, all drawings are stored as a set of strokes. 
  For info on raw format see: https://github.com/googlecreativelab/quickdraw-dataset
- This script converts the strokes of 2000 drawings for the countries US, RU and DE RGB images 
  with black strokes on white background, one in original size (256x256) and reduced size (32x32)
  
For one or ALL files in the data/ directory:
- Read json file of a word (in filename) or ALL files in directory
- Keep only the drawings that were recognized 
- For the countries US, RU and DE:
    - Extract 2000 drawings for the given country 
    - Turn strokes into RGB image of original size (256x256)
    - Turn strokes into RGB image of reduced size (32x32)
    - Append word, country code, img_256, img_32 to data frame
- Save dataframe as .npy to preseve array structure 

Input: 
- -w, --word, str, name of a word (corresponding file of full-simplified-{word}.ndjson, stored in ../data/) or "ALL"

Returns: 
- Saves .npy file for the word or ALL in ../out/0_preprocessed_data/
"""


# LIBRARIES ---------------------------------------------------

# Basics
import os
import glob
import ujson
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

# Image processing
import cv2


# HELPER FUNCTION --------------------------------------------    
        
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
    # Resize to output size
    img_sized = cv2.resize(img_rgb, (output_size, output_size), interpolation = cv2.INTER_AREA)
            
    return img_sized


# MAIN FUNCTION -----------------------------------------------

def main():
    
    # --- ARGUMENT PARSER AND OUTPUT DIRECTORY ---
    
    # Argument parser for word input
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--word", type = str, required = True)
    args = vars(ap.parse_args())
    word = args["word"]
    
    # Prepare output directory
    output_directory = os.path.join("..", "out", "0_preprocessed_data")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # --- GET FILEPATHS ---
    
    # Print messsage
    print(f"\n[INFO] Initializing preprocessing of {word}!")
    
    # Get paths: if "ALL" get all paths of .ndjson files, otherwise only path for word
    if (word == "ALL"):
        path_list = glob.glob(os.path.join("..", "data", "*.ndjson"))
    else: 
        path_list = [os.path.join("..", "data", f"full-simplified-{word}.ndjson")]      
        
    # --- PREPROCESS DATA ---
        
    # For each path in the list
    for path in tqdm(path_list):
        
        # Load json file and turn to data frame
        json = map(ujson.loads, open(path))
        df = pd.DataFrame.from_records(json)
        
        # Select only the recognized drawings
        df = df[df["recognized"] == True]

        # Create empty target dataframe to save word, country and drawing in 256x256 and 32x32
        df_out = pd.DataFrame(columns = ["word", "country", "img_256", "img_32"])

        # For each country in the relevant countries
        for country in (["US", "RU", "DE"]): 

            # Select only rows for the given country and take first 2000 rows
            df_sample = df[df["countrycode"] == country][:2000]

            # For each row
            for index, row in df_sample.iterrows():
                # Get word, replace space with _
                pretty_word = row["word"].replace(" ", "_")
                # Get country code
                country = row["countrycode"]
                # Get images of original size: 256x256
                img_256 = strokes_to_img(row["drawing"], 256)
                # Get images of reduced size: 32x32
                img_32 = strokes_to_img(row["drawing"], 32)
                # Append all elements to dataframe
                df_out = df_out.append({"word": pretty_word,
                                        "country": country,
                                        "img_256": np.array(img_256),
                                        "img_32": np.array(img_32)}, ignore_index=True)

        # Save out dataframe in output directory dataframe as .npy 
        out_path = os.path.join(output_directory, f"{pretty_word}.npy")
        np.save(out_path, df_out.to_numpy())

    # Print messsage
    print(f"[INFO] All done, output saved in {output_directory}!")
                                
                                                           
if __name__=="__main__":
    main()