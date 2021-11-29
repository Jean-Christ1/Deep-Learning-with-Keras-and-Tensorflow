<a name="top"></a>

[<img width="600px" src="fidle/img/00-Fidle-titre-01.svg"></img>](#top)

<!-- --------------------------------------------------- -->
<!-- To correctly view this README under Jupyter Lab     -->
<!-- Open the notebook: README.ipynb!                    -->
<!-- --------------------------------------------------- -->

## About Fidle

This repository contains all the documents and links of the **Fidle Training** .   
Fidle (for Formation Introduction au Deep Learning) is a 2-day training session  
co-organized by the Formation Permanente CNRS and the Resinfo/SARI and DevLOG CNRS networks.  

The objectives of this training are :
 - Understanding the **bases of Deep Learning** neural networks
 - Develop a **first experience** through simple and representative examples
 - Understanding **Tensorflow/Keras** and **Jupyter lab** technologies
 - Apprehend the **academic computing environments** Tier-2 or Tier-1 with powerfull GPU

For more information, see **https://fidle.cnrs.fr** :
- **[Fidle site](https://fidle.cnrs.fr)**
- **[Presentation of the training](https://fidle.cnrs.fr/presentation)**
- **[Program 2021/2022](https://fidle.cnrs.fr/programme)**
- [Subscribe to the list](https://fidle.cnrs.fr/listeinfo), to stay informed !
- [Find us on youtube](https://fidle.cnrs.fr/youtube)

For more information, you can contact us at :  
[<img width="200px" style="vertical-align:middle" src="fidle/img/00-Mail_contact.svg"></img>](#top)

Current Version : <!-- VERSION_BEGIN -->
**2.0.29**
<!-- VERSION_END -->


## Course materials

| | | | |
|:--:|:--:|:--:|:--:|
| **[<img width="50px" src="fidle/img/00-Fidle-pdf.svg"></img><br>Course slides](https://fidle.cnrs.fr/supports)**<br>The course in pdf format<br>(12 Mo)| **[<img width="50px" src="fidle/img/00-Notebooks.svg"></img><br>Notebooks](https://fidle.cnrs.fr/notebooks)**<br> &nbsp;&nbsp;&nbsp;&nbsp;Get a Zip or clone this repository &nbsp;&nbsp;&nbsp;&nbsp;<br>(40 Mo)| **[<img width="50px" src="fidle/img/00-Datasets-tar.svg"></img><br>Datasets](https://fidle.cnrs.fr/fidle-datasets.tar)**<br>All the needed datasets<br>(1.2 Go)|**[<img width="50px" src="fidle/img/00-Videos.svg"></img><br>Videos](https://fidle.cnrs.fr/youtube)**<br>&nbsp;&nbsp;&nbsp;&nbsp;Our Youtube channel&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;|

Have a look about **[How to get and install](https://fidle.cnrs.fr/installation)** these notebooks and datasets.


## Jupyter notebooks

<!-- INDEX_BEGIN -->

### Linear and logistic regression
- **[LINR1](LinearReg/01-Linear-Regression.ipynb)** - [Linear regression with direct resolution](LinearReg/01-Linear-Regression.ipynb)  
Low-level implementation, using numpy, of a direct resolution for a linear regression
- **[GRAD1](LinearReg/02-Gradient-descent.ipynb)** - [Linear regression with gradient descent](LinearReg/02-Gradient-descent.ipynb)  
Low level implementation of a solution by gradient descent. Basic and stochastic approach.
- **[POLR1](LinearReg/03-Polynomial-Regression.ipynb)** - [Complexity Syndrome](LinearReg/03-Polynomial-Regression.ipynb)  
Illustration of the problem of complexity with the polynomial regression
- **[LOGR1](LinearReg/04-Logistic-Regression.ipynb)** - [Logistic regression](LinearReg/04-Logistic-Regression.ipynb)  
Simple example of logistic regression with a sklearn solution

### Perceptron Model 1957
- **[PER57](IRIS/01-Simple-Perceptron.ipynb)** - [Perceptron Model 1957](IRIS/01-Simple-Perceptron.ipynb)  
Example of use of a Perceptron, with sklearn and IRIS dataset of 1936 !

### Basic regression using DNN
- **[BHPD1](BHPD/01-DNN-Regression.ipynb)** - [Regression with a Dense Network (DNN)](BHPD/01-DNN-Regression.ipynb)  
Simple example of a regression with the dataset Boston Housing Prices Dataset (BHPD)
- **[BHPD2](BHPD/02-DNN-Regression-Premium.ipynb)** - [Regression with a Dense Network (DNN) - Advanced code](BHPD/02-DNN-Regression-Premium.ipynb)  
A more advanced implementation of the precedent example

### Basic classification using a DNN
- **[MNIST1](MNIST/01-DNN-MNIST.ipynb)** - [Simple classification with DNN](MNIST/01-DNN-MNIST.ipynb)  
An example of classification using a dense neural network for the famous MNIST dataset
- **[MNIST2](MNIST/02-CNN-MNIST.ipynb)** - [Simple classification with CNN](MNIST/02-CNN-MNIST.ipynb)  
An example of classification using a convolutional neural network for the famous MNIST dataset

### Images classification with Convolutional Neural Networks (CNN)
- **[GTSRB1](GTSRB/01-Preparation-of-data.ipynb)** - [Dataset analysis and preparation](GTSRB/01-Preparation-of-data.ipynb)  
Episode 1 : Analysis of the GTSRB dataset and creation of an enhanced dataset
- **[GTSRB2](GTSRB/02-First-convolutions.ipynb)** - [First convolutions](GTSRB/02-First-convolutions.ipynb)  
Episode 2 : First convolutions and first classification of our traffic signs
- **[GTSRB3](GTSRB/03-Tracking-and-visualizing.ipynb)** - [Training monitoring](GTSRB/03-Tracking-and-visualizing.ipynb)  
Episode 3 : Monitoring, analysis and check points during a training session
- **[GTSRB4](GTSRB/04-Data-augmentation.ipynb)** - [Data augmentation ](GTSRB/04-Data-augmentation.ipynb)  
Episode 4 : Adding data by data augmentation when we lack it, to improve our results
- **[GTSRB5](GTSRB/05-Full-convolutions.ipynb)** - [Full convolutions](GTSRB/05-Full-convolutions.ipynb)  
Episode 5 : A lot of models, a lot of datasets and a lot of results.
- **[GTSRB6](GTSRB/06-Notebook-as-a-batch.ipynb)** - [Full convolutions as a batch](GTSRB/06-Notebook-as-a-batch.ipynb)  
Episode 6 : To compute bigger, use your notebook in batch mode
- **[GTSRB7](GTSRB/07-Show-report.ipynb)** - [Batch reports](GTSRB/07-Show-report.ipynb)  
Episode 7 : Displaying our jobs report, and the winner is...
- **[GTSRB10](GTSRB/batch_oar.sh)** - [OAR batch script submission](GTSRB/batch_oar.sh)  
Bash script for an OAR batch submission of an ipython code
- **[GTSRB11](GTSRB/batch_slurm.sh)** - [SLURM batch script](GTSRB/batch_slurm.sh)  
Bash script for a Slurm batch submission of an ipython code

### Sentiment analysis with word embedding
- **[IMDB1](IMDB/01-One-hot-encoding.ipynb)** - [Sentiment analysis with hot-one encoding](IMDB/01-One-hot-encoding.ipynb)  
A basic example of sentiment analysis with sparse encoding, using a dataset from Internet Movie Database (IMDB)
- **[IMDB2](IMDB/02-Keras-embedding.ipynb)** - [Sentiment analysis with text embedding](IMDB/02-Keras-embedding.ipynb)  
A very classical example of word embedding with a dataset from Internet Movie Database (IMDB)
- **[IMDB3](IMDB/03-Prediction.ipynb)** - [Reload and reuse a saved model](IMDB/03-Prediction.ipynb)  
Retrieving a saved model to perform a sentiment analysis (movie review)
- **[IMDB4](IMDB/04-Show-vectors.ipynb)** - [Reload embedded vectors](IMDB/04-Show-vectors.ipynb)  
Retrieving embedded vectors from our trained model
- **[IMDB5](IMDB/05-LSTM-Keras.ipynb)** - [Sentiment analysis with a RNN network](IMDB/05-LSTM-Keras.ipynb)  
Still the same problem, but with a network combining embedding and RNN

### Time series with Recurrent Neural Network (RNN)
- **[LADYB1](SYNOP/LADYB1-Ladybug.ipynb)** - [Prediction of a 2D trajectory via RNN](SYNOP/LADYB1-Ladybug.ipynb)  
Artificial dataset generation and prediction attempt via a recurrent network
- **[SYNOP1](SYNOP/SYNOP1-Preparation-of-data.ipynb)** - [Preparation of data](SYNOP/SYNOP1-Preparation-of-data.ipynb)  
Episode 1 : Data analysis and preparation of a usuable meteorological dataset (SYNOP)
- **[SYNOP2](SYNOP/SYNOP2-First-predictions.ipynb)** - [First predictions at 3h](SYNOP/SYNOP2-First-predictions.ipynb)  
Episode 2 : RNN training session for weather prediction attempt at 3h
- **[SYNOP3](SYNOP/SYNOP3-12h-predictions.ipynb)** - [12h predictions](SYNOP/SYNOP3-12h-predictions.ipynb)  
Episode 3: Attempt to predict in a more longer term 

### Unsupervised learning with an autoencoder neural network (AE)
- **[AE1](AE/01-Prepare-MNIST-dataset.ipynb)** - [Prepare a noisy MNIST dataset](AE/01-Prepare-MNIST-dataset.ipynb)  
Episode 1: Preparation of a noisy MNIST dataset
- **[AE2](AE/02-AE-with-MNIST.ipynb)** - [Building and training an AE denoiser model](AE/02-AE-with-MNIST.ipynb)  
Episode 1 : Construction of a denoising autoencoder and training of it with a noisy MNIST dataset.
- **[AE3](AE/03-AE-with-MNIST-post.ipynb)** - [Playing with our denoiser model](AE/03-AE-with-MNIST-post.ipynb)  
Episode 2 : Using the previously trained autoencoder to denoise data
- **[AE4](AE/04-ExtAE-with-MNIST.ipynb)** - [Denoiser and classifier model](AE/04-ExtAE-with-MNIST.ipynb)  
Episode 4 : Construction of a denoiser and classifier model
- **[AE5](AE/05-ExtAE-with-MNIST.ipynb)** - [Advanced denoiser and classifier model](AE/05-ExtAE-with-MNIST.ipynb)  
Episode 5 : Construction of an advanced denoiser and classifier model

### Generative network with Variational Autoencoder (VAE)
- **[VAE1](VAE/01-VAE-with-MNIST.ipynb)** - [First VAE, using functional API (MNIST dataset)](VAE/01-VAE-with-MNIST.ipynb)  
Construction and training of a VAE, using functional APPI, with a latent space of small dimension.
- **[VAE2](VAE/02-VAE-with-MNIST.ipynb)** - [VAE, using a custom model class  (MNIST dataset)](VAE/02-VAE-with-MNIST.ipynb)  
Construction and training of a VAE, using model subclass, with a latent space of small dimension.
- **[VAE3](VAE/03-VAE-with-MNIST-post.ipynb)** - [Analysis of the VAE's latent space of MNIST dataset](VAE/03-VAE-with-MNIST-post.ipynb)  
Visualization and analysis of the VAE's latent space of the dataset MNIST
- **[VAE5](VAE/05-About-CelebA.ipynb)** - [Another game play : About the CelebA dataset](VAE/05-About-CelebA.ipynb)  
Episode 1 : Presentation of the CelebA dataset and problems related to its size
- **[VAE6](VAE/06-Prepare-CelebA-datasets.ipynb)** - [Generation of a clustered dataset](VAE/06-Prepare-CelebA-datasets.ipynb)  
Episode 2 : Analysis of the CelebA dataset and creation of an clustered and usable dataset
- **[VAE7](VAE/07-Check-CelebA.ipynb)** - [Checking the clustered dataset](VAE/07-Check-CelebA.ipynb)  
Episode : 3 Clustered dataset verification and testing of our datagenerator
- **[VAE8](VAE/08-VAE-with-CelebA.ipynb)** - [Training session for our VAE](VAE/08-VAE-with-CelebA.ipynb)  
Episode 4 : Training with our clustered datasets in notebook or batch mode
- **[VAE9](VAE/09-VAE-with-CelebA-post.ipynb)** - [Data generation from latent space](VAE/09-VAE-with-CelebA-post.ipynb)  
Episode 5 : Exploring latent space to generate new data
- **[VAE10](VAE/batch_slurm.sh)** - [SLURM batch script](VAE/batch_slurm.sh)  
Bash script for SLURM batch submission of VAE8 notebooks 

### Generative Adversarial Networks (GANs)
- **[DCGAN01](DCGAN/01-DCGAN-Draw-me-a-sheep.ipynb)** - [A first DCGAN to Draw a Sheep](DCGAN/01-DCGAN-Draw-me-a-sheep.ipynb)  
Episode 1 : Draw me a sheep, revisited with a DCGAN

### Miscellaneous
- **[ACTF1](Misc/Activation-Functions.ipynb)** - [Activation functions](Misc/Activation-Functions.ipynb)  
Some activation functions, with their derivatives.
- **[NP1](Misc/Numpy.ipynb)** - [A short introduction to Numpy](Misc/Numpy.ipynb)  
Numpy is an essential tool for the Scientific Python.
- **[SCRATCH1](Misc/Scratchbook.ipynb)** - [Scratchbook](Misc/Scratchbook.ipynb)  
A scratchbook for small examples
- **[TSB1](Misc/Using-Tensorboard.ipynb)** - [Tensorboard with/from Jupyter ](Misc/Using-Tensorboard.ipynb)  
4 ways to use Tensorboard from the Jupyter environment
<!-- INDEX_END -->


## Installation

Have a look about **[How to get and install](https://fidle.cnrs.fr/installation)** these notebooks and datasets.

## Licence

[<img width="100px" src="fidle/img/00-fidle-CC BY-NC-SA.svg"></img>](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
\[en\] Attribution - NonCommercial - ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
\[Fr\] Attribution - Pas d’Utilisation Commerciale - Partage dans les Mêmes Conditions 4.0 International  
See [License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
See [Disclaimer](https://creativecommons.org/licenses/by-nc-sa/4.0/#).  


----
[<img width="80px" src="fidle/img/00-Fidle-logo-01.svg"></img>](#top)
