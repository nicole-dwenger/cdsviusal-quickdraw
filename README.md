# QuickDraw! - Culture in Sketches

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
>This project is the self-assigned project of the course Visual Analytics. 


The aim of this project was to investigate sketches that were collected by the QuickDraw application from Google. In the [QuickDraw application](https://quickdraw.withgoogle.com), users are prompted with a word and asked to draw the given concept. While drawing, Google’s classification model tries to guess the corresponding word. These quick drawings might provide insights into how people represent words or concepts in a very simple format. Many approaches have been taken to classify the words of drawings. This project aimed to go beyond this classification task, by also taking a cultural approach. Specifically, it was investigated whether it is possible to predict the country a drawing came from. People from different countries might have different representations of e.g beard, sandwich or rain as a consequence of their surrounding and cultural background. Further, an exploratory and unsupervised approach was taken to investigate, whether any other clusters can be identified within the drawings of a word, as people across cultures may have different ways of representing words visually. Thus, the following questions were posed: 

1. Can the word of a drawings be classified? 
2. Can the country (of the artist) of drawings of a word be classified?
3. Exploratory + unsupervised: Can any other clusters be identified within the drawings of a single word?

For this project, drawings of the following 10 words were used: *beard, birthday cake, face, house, ice cream, rain, sandwich, snowflake, The Mona Lisa, yoga*. Further, to reduce complexity, only drawings from Germany (DE), Russia (RU) and the United States (US) were considered. The motivation for choosing these drawings was the amount of data and that all countries are on different continents and have different native languages. 

## Methods
### Data and Preprocessing
For each of the 10 selected words *beard, birthday cake, face, house, ice cream, rain, sandwich, snowflake, The Mona Lisa, yoga* the simplified drawings were extracted from the [database](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) provided by Google as .ndjson files. In these simplified drawing files, drawings are aligned to the top left corner and scaled to be of size 256x256 (further documentation [here](https://github.com/googlecreativelab/quickdraw-dataset#get-the-data)). However, drawings in these files are represented by their single stokes. Thus, it was necessary to transform the strokes into images. Specifically, the following steps were taken to preprocess each of the 10 .ndjson files:

1. Filter out drawings which were not recognized by the Google classifier.
2. Extract the first 2000 images for each of the three countries (DE, RU, US)
3. Transform strokes into an image by drawing lines onto an empty array of 256x256.
4. Convert grey images to RGB scale for VGG16, using `cv2.COLOR_GRAY2RGB`. 
5. Save RGB image of drawing in original size (256x256x3)
6. Save RGB image of drawing in reduced size (32x32x3)
7. Save data as .npy file with columns of word, country, img_256, img_32.

Thus, in total 60.000 drawings were preprocessed, 6.000 for each of the 10 categories, of which 2000 belonged to each of the three countries. Here are some examples of the preprocessed drawings for the 10 categories:

![](https://github.com/nicole-dwenger/cdsviusal-quickdraw/blob/master/out/assets/examples.png)

### Transfer Learning 
For this project, the method of transfer learning was applied. Specifically, pre-trained layers of the model VGG16 were used, while fully connected layers at the end (or in tensorflow-terms *at the top*) were removed and substituted with layers fitting to the data used in this project. Transfer learning is useful, as the model (here VGG16) has been trained on a large collection of images and learned to *see*, meaning to extract features from images. Thus, only the fully connected layers at the end are being trained, reducing processing time and required amount of data. 

#### 1. Can the word that was presented in relation to the drawing be classified? 
For this classification task, all 60.000 preprocessed drawing of size 32x32 or the 10 words were normalised using min-max-regularisation and split into train and test images using an 80/20 split. To the pre-trained layers of VGG16, a flattening layer, dense layer with 256 nodes and an output layer to classify the 10 words were appended. As optimiser Adam was used with a learning rate of 0.001 and the model was trained for 5 epochs using batch size of 40. 

#### 2. Can the country of a drawings belonging to a single word be classified? 
For this classification task, drawings of one word were considered at a time. Thus, 10 models were trained on a total of 6.000 images (2000 for each country). These images were of size 32x32x3, regularised with min-max-regularisation and split into test and training data using an 80/20 split. To the pre-trained layers of VGG16 a flattening layer, dense layer with 256 nodes, a drop out layer with a drop out rate of 0.02 to avoid overfitting and an output layer to classify the 3 countries were appended. As optimiser Adam was used with a learning rate of 0.001 and each model was trained for 20 epochs using batch size of 40. Different batch sizes and epochs were explored, but increased overfitting. 

#### 3. Can any other clusters be identified within the drawings of a word? 
For this unsupervised, exploratory task, pre-trained weights of VGG16 were used to extract dense feature representations (of size 512) for each of the drawings belonging to one word. These dense feature representations were normalised and fed into a K-Means clustering algorithm. For all words, k was defined to be 5. Note, that this value was not an informed choice but rather based on intuition. Using the cluster labels assigned to each image, 20 images were sampled for each cluster, plotted and saved as an image.


## Repository Structure
```
|-- data/                              # Directory of raw data, only containing sample of files
    |-- full-simplified-beard.ndjson   # Simplified drawing file, one for each word, only three examples here
    |-- full-simplified-face.ndjson
    |-- ...

|-- out/                               # Directory containing example output
    |-- 0_preprocessed_data/           # Preprocessed drawings, as .npy files, one for each word, empty as files are too large
    |-- 1_word_classification/         # Model output of 1_word_classification.py
    |-- 2_country_classification/      # Model output of 2_country_classification.py, one for each word
    |-- 3_clustering/                  # Example images of clusters, one .png for each word

|-- src/                               # Directory containing the main scripts
    |-- 0_preprocess.py                # Preprocessinng of .ndjson files to .npy files
    |-- 1_word_classification.py       # Classification of words of all preprocessed drawings
    |-- 2_country_classification.py    # Classification of country (DE, RU, US) of drawings of one word
    |-- 3_clustering.py                # Unsupervised, kmeans clustering of drawings of one word, using feature extraction

|-- utils/                             # Directory containing utility script
    |-- quickdraw_utils.py             # Script, containing functions sourced across the main scripts

|-- create_venv.sh                     # Bash script to create virtual environment, venv_quickdraw
|-- requirements.txt                   # Necessary dependencies to run script
|-- README.md   

```

## Usage
**!** The scripts have only been tested on Linux, using Python 3.6.9.

### 1. Cloning the Repository and Installing Dependencies

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_quickdraw` with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdsvisual-quickdraw.git

# move into directory
cd cdsvisual-quickdraw/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_quickdraw/bin/activate
```

### 2. Data
The simplified .ndjson drawing files, which were used in this project can be downloaded from [Google](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). In the `data/` repository I have provided files of 3 words, which were small enough to store on GitHub. If you wish to reproduce all results of this project, the reamining .ndjson files should be downloaded and saved in the `data/` directory.  

### 3. Scripts
This repository contains three main scripts in the `src/` directory, some of which also source functions from the utility script `utils/quickdraw_utils.py`. Note, that the scripts `1_word_classification.py`, `2_country_classification.py` and `3_clustering.py` require that raw data (.ndjson files) were processed with `0_preprocessing.py`. Detailed descriptions how to run each of them are provided below. Example output can be found in the corresponding `out/` directories.

### 3.0. Preprocessing: 0_preprocessing.py
The script `0_preprocessing.py` preprocesses .ndjson files stored in the `data/` directory as described above. The script should be called after directing to the `src/` directory:

```bash
# moving into src
cd src/

# running script for one word
python3 0_preprocessing.py -w yoga

# running script for .ndjson all files in data/
python3 0_preprocessing.py -w ALL
```

__Parameters:__
- `-w, --word`: *str*, ***required***\
   Word, for which a corresponding .ndjson file is stored in `data/`. Note, that the word should be written in the same way as it is in the filename, e.g., for `full-simplified-The Mona Lisa.ndjson` this would be `The Mona Lisa`. Use `ALL`, if all files should be processed at once. 

__Output__ saved in `out/0_preprocessing/`:
- `{word}.npy`\
   Preprocessed data, corresponding to one .ndjson file. Stored as .npy file to preserve arrays, but is arranged in columns of `word, country, img_256, img_32`. Note, that if word labels previously had spaces in the file name, they have been replaced with an underscore `_`, to prevent any issues with filenames.

### 3.1. Classification of words: 1_word_classification.py
The script `1_word_classification.py` takes all .npy files stored in the `out/0_preprocessed_data` directory, and uses all drawings to classify their belonging words. Note, that this script is a multi-class classifier and thus requires there to be more than 2 files (i.e. drawings of more than 2 words). The script should be called after directing to the `src/` directory:

```bash
# moving into src
cd src/

# running script for one word
python3 1_word_classification.py 

# running script for .ndjson all files in data/
python3 1_word_classification.py -e 20 
```

__Parameters:__
- `-b, --batch_size`: *int, optional, default:* `40`\
   Batch size to chunk data into when training model. 

- `-e, --epochs`: *int, optional, default:* `5`\
   Number of epochs to train model for.  

__Output__ saved in `out/1_word_classification/`:
- `model_summary.txt`\
    Summary of model architecture and number of parameters. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_plot.png`\
   Visualisation of model architecture, i.e. layers. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_history.png`\
   Training history of model, i.e. training and validation loss and accuracy over epochs. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_report.txt`\
   Classification report of the model. If the file exists already, a number will be added to the filename to avoid overwriting. Also printed to command line.


### 3.2. Classification of countries: 2_country_classification.py
The script `2_word_classification.py` takes one .npy file stored in the `out/0_preprocessed_data` directory to train the model to classify drawings as belonging to one of the three countries (DE, RU, US). The script should be called after directing to the `src/` directory:

```bash
# moving into src
cd src/

# running script for specified word
python3 2_country_classification.py -w The_Mona_Lisa
```

__Parameters:__
- `-w, --word`: *str*, ***required***\
   Word, for which a corresponding .npy file is stored in `out/0_preprocessed_data`. Note, that the word should be written in the same way as it is in the filename, e.g., for a file called The_Mona_Lisa.npy this would be `The_Mona_Lisa`.
   
- `-b, --batch_size`: *int, optional, default:* `40`\
   Batch size to chunk data into when training model. 

- `-e, --epochs`: *int, optional, default:* `20`\
   Number of epochs to train model for.  

__Output__ saved in `out/2_country_classification/{word}`:
- `model_summary.txt`\
    Summary of model architecture and number of parameters. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_plot.png`\
   Visualisation of model architecture, i.e. layers. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_history.png`\
   Training history of model, i.e. training and validation loss and accuracy over epochs. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_report.txt`\
   Classification report of the model. If the file exists already, a number will be added to the filename to avoid overwriting. Also printed to command line.
   
   
### 3.2. Unsupervised clustering of drawings of a word: 3_clustering.py
The script `3_clustering.py` takes one .npy file stored in the `out/0_preprocessed_data` directory to extract features of all drawings, generate k clusters and plot 20 examples of each cluster on an image. The script should be called after directing to the `src/` directory:

```bash
# moving into src
cd src/

# running script for one word
python3 3_clustering.py -w yoga 
```

__Parameters:__
- `-w, --word`: *str*, ***required***\
   Word, for which a corresponding .npy file is stored in `out/0_preprocessed_data`. Note, that the word should be written in the same way as it is in the filename, e.g., for a file called The_Mona_Lisa.npy this would be `The_Mona_Lisa`.
   
- `-k, --k_clusters`: *int, optional, default:* `5`\
   Number of k clusters for kmeans algorithm. 
  

__Output__ saved in `out/3_clustering/`:
- `{word}_{k_clusters}_clusters.png`\
   Image with 20 examples of each of the clusters. 

## Results and Discussion

### 1. Can the word that was presented in relation to the sketch be classified? 
The model, which was trained to classify drawings as belonging to one of the 10 words reached an F1 score of 0.92. The model history plot indicates, that already from the first epoch, the model achieved a quite high training and validation accuracy and did not improve much over the 5 epochs.  

### 2. Can the country that the artist of a drawing for a single word be classified? 
All outputs of the country classification can be found in `out/2_country_classification/` directory. The following F1 scores were achieved when training a model to classify the country (DE, RU, US) within drawings belonging to one of the 10 words: 

| word | DE F1 | RU F1 | US F1 | weighted overall F1 
|--|--|--|--|--|
| The Mona Lisa | 0.44 | 0.46 | 0.36  | 0.43 |
| beard | 0.43 | 0.40 | 0.42 | 0.42 |
| birthday cake | 0.38 | 0.24 | 0.50 | 0.37 |
| face | 0.34 | 0.43 | 0.51 |  0.42 |
| house | 0.41 | 0.53 | 0.51 |  0.49 |
| ice_cream | 0.50 | 0.39 | 0.63 |  0.51 |
| rain | 0.46 | 0.40 | 0.43 |  0.43 |
| sandwich | 0.30 | 0.52 | 0.39 |0.40 | 
| snowflake | 0.39 | 0.46 | 0.36 | 0.40 | 
| yoga | 0.20 | 0.54 | 0.52 | 0.42 | 

For none of these words it was possible to reliably classify the country of a drawing. Plots of the model history indicated, that many of the models did not improve over epochs. In some cases only the training accuracy started to improve, suggesting that the model was starting to overfit on the training data. Implications and critical reflections are addressed in the discussion below.  
### 3. Can any other clusters be identified from drawings belonging to the same word? 
Images were clustered into 5 clusters using features extracted from VGG16. Plots for all of the 10 words can be found in the `out/3_clustering/` directory. For some words, it is possible to see some differences between clusters (e.g. rain, snowflake, yoga). For other words (e.g. house, ice cream, beard) from simply looking at the examples, it does not seem like clusters can be clearly distinguished. Examples are provided below. 

In the future, it should be considered, to explore a range of possible values for k. Further, instead of sampling images for each cluster, those which are closest to the centroid of each cluster could be plotted. Nevertheless, what these plots also indicate is also that the drawings are noisy and not always centred in the image, which might have contributed to the fact that they could not be classified by their country. Further implications are discussed below.

__Clusters for *rain*:__ rain ...\
0: ... as lines, 1: ... as more lines, 2: .. with clouds at the top, 3: ... with bigger clouds, 4: ... with clouds and big drops
![](https://github.com/nicole-dwenger/cdsviusal-quickdraw/blob/master/out/3_clustering/rain_5_clusters.png)

__Clusters for *snowflake*:__ snowflake ...\
0: ... as simple lines, 1: ... as blob - 2: ... as lines at the top, 3: ... as fewer lines - 4: ... as decorated lines
![](https://github.com/nicole-dwenger/cdsviusal-quickdraw/blob/master/out/3_clustering/snowflake_5_clusters.png)

__Clusters for *yoga*:__\
0: cheating with writing - 1: yoga with spread out legs/arms - 2: standing yoga - 3: yoga on mat - 4: cut off drawing
![](https://github.com/nicole-dwenger/cdsviusal-quickdraw/blob/master/out/3_clustering/yoga_5_clusters.png)

### 3. Discussion 
There are several aspects which should be considered critically in relation to this project. First, the words, for which drawings were used in this project may have been too simple. In other words, if the word is simple, such as *house*, many people may have a very similar representation of it, making it difficult to distinguish between countries. Similarly, the countries, for which drawings were chosen may not have been different enough in the way they represent words visually.
Further, the images were transformed to be on RGB scale, even though they were grey scale. This did not seem to impede classification of sketches by word. However, it should still be mentioned, that VGG16 is trained on coloured images, and in this project it was pretended that images were coloured. Additionally, the sketches are very different from the images of ImageNet, which VGG16 was trained on. Thus, the model might not be fitting for the kind of simple sketches used in this project.
Lastly, there are some general issues with the QuickDraw data. For instance, the users were stropped from drawing as soon as the classifier from Google recognized the sketch. Thus, the sketches may be incomplete and inaccurate representations. Further, the sketches were drawn on a computer. This may make drawings more messy and induce noise in the data.

## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk.