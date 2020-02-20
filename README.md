<!-- ![](fidle/img/00-Fidle-titre-01_m.png) -->
[<img width="600px" src="fidle/img/00-Fidle-titre-01.svg"></img>](#)

## A propos

This repository contains all the documents and links of the **Fidle Training**.  

The objectives of this training, co-organized by the Formation Permanente CNRS and the SARI and DEVLOG networks, are :
 - Understanding the **bases of deep learning** neural networks (Deep Learning)
 - Develop a **first experience** through simple and representative examples
 - Understand the different types of networks, their **architectures** and their **use cases**.
 - Understanding **Tensorflow/Keras and Jupyter lab** technologies on the GPU
 - Apprehend the **academic computing environments** Tier-2 (meso) and/or Tier-1 (national)

## Course materials
Get the **[course slides](Bientot)**  
<img width="50px" src="fidle/img/00-Fidle-pdf.svg"></img>


<!-- ![pdf](fidle/img/00-Fidle-pdf.png) -->
Useful information is also available in the [wiki](https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/wikis/home)


**Jupyter notebooks :**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgricad-gitlab.univ-grenoble-alpes.fr%2Ftalks%2Fdeeplearning.git/master?urlpath=lab/tree/index.ipynb)


<!-- DO NOT REMOVE THIS TAG !!! -->
<!-- INDEX -->
<!-- INDEX_BEGIN -->
1. [Linear regression with direct resolution](LinearReg/01-Linear-Regression.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Direct determination of linear regression 
1. [Linear regression with gradient descent](LinearReg/02-Gradient-descent.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;An example of gradient descent in the simple case of a linear regression.
1. [Complexity Syndrome](LinearReg/03-Polynomial-Regression.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Illustration of the problem of complexity with the polynomial regression
1. [Logistic regression, in pure Tensorflow](LinearReg/04-Logistic-Regression.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Logistic Regression with Mini-Batch Gradient Descent using pure TensorFlow. 
1. [Regression with a Dense Network (DNN)](BHPD/01-DNN-Regression.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A Simple regression with a Dense Neural Network (DNN) - BHPD dataset
1. [Regression with a Dense Network (DNN) - Advanced code](BHPD/02-DNN-Regression-Premium.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;More advanced example of DNN network code - BHPD dataset
1. [CNN with GTSRB dataset - Data analysis and preparation](GTSRB/01-Preparation-of-data.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 1: Data analysis and creation of a usable dataset
1. [CNN with GTSRB dataset - First convolutions](GTSRB/02-First-convolutions.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 2 : First convolutions and first results
1. [CNN with GTSRB dataset - Monitoring ](GTSRB/03-Tracking-and-visualizing.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 3: Monitoring and analysing training, managing checkpoints
1. [CNN with GTSRB dataset - Data augmentation ](GTSRB/04-Data-augmentation.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 4: Improving the results with data augmentation
1. [CNN with GTSRB dataset - Full convolutions ](GTSRB/05-Full-convolutions.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 5: A lot of models, a lot of datasets and a lot of results.
1. [CNN with GTSRB dataset - Full convolutions as a batch](GTSRB/06-Full-convolutions-batch.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 6 : Run Full convolution notebook as a batch
1. [Tensorboard with/from Jupyter ](GTSRB/99-Scripts-Tensorboard.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4 ways to use Tensorboard from the Jupyter environment
1. [Text embedding with IMDB](IMDB/01-Embedding-Keras.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A very classical example of word embedding for text classification (sentiment analysis)
1. [Text embedding with IMDB - Reloaded](IMDB/02-Prediction.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example of reusing a previously saved model
1. [Text embedding/LSTM model with IMDB](IMDB/03-LSTM-Keras.ipynb)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Still the same problem, but with a network combining embedding and LSTM
<!-- INDEX_END -->



## Installation
To run this examples, you need an environment with the following packages :
 - Python >3.5
 - numpy
 - Tensorflow 2.0
 - scikit-image
 - scikit-learn
 - Matplotlib
 - seaborn
 - pyplot

You can install such a predefined environment :
```
conda env create -f environment.yml
```

To manage conda environment see [there](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)  



## Licence

\[en\] Attribution - NonCommercial - ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
\[Fr\] Attribution - Pas d’Utilisation Commerciale - Partage dans les Mêmes Conditions 4.0 International
See [License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
See [Disclaimer](https://creativecommons.org/licenses/by-nc-sa/4.0/#).