<a name="top"></a>

[<img width="600px" src="fidle/img/00-Fidle-titre-01.svg"></img>](#top)

<!-- --------------------------------------------------- -->
<!-- To correctly view this README under Jupyter Lab     -->
<!-- Open the notebook: README.ipynb!                    -->
<!-- --------------------------------------------------- -->


## A propos

This repository contains all the documents and links of the **Fidle Training** .   
Fidle (for Formation Introduction au Deep Learning) is a 2-day training session  
co-organized by the Formation Permanente CNRS and the SARI and DEVLOG networks.  

The objectives of this training are :
 - Understanding the **bases of Deep Learning** neural networks
 - Develop a **first experience** through simple and representative examples
 - Understanding **Tensorflow/Keras** and **Jupyter lab** technologies
 - Apprehend the **academic computing environments** Tier-2 or Tier-1 with powerfull GPU

For more information, you can contact us at : 
[<img width="200px" style="vertical-align:middle" src="fidle/img/00-Mail_contact.svg"></img>](#top)  
Current Version : <!-- VERSION_BEGIN -->
1.2b1 DEV
<!-- VERSION_END -->


## Course materials

| | | |
|:--:|:--:|:--:|
| **[<img width="50px" src="fidle/img/00-Fidle-pdf.svg"></img><br>Course slides](https://cloud.univ-grenoble-alpes.fr/index.php/s/wxCztjYBbQ6zwd6)**<br>The course in pdf format<br>(12 Mo)| **[<img width="50px" src="fidle/img/00-Notebooks.svg"></img><br>Notebooks](https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/archive/master/fidle-master.zip)**<br> &nbsp;&nbsp;&nbsp;&nbsp;Get a Zip or clone this repository &nbsp;&nbsp;&nbsp;&nbsp;<br>(10 Mo)| **[<img width="50px" src="fidle/img/00-Datasets-tar.svg"></img><br>Datasets](https://cloud.univ-grenoble-alpes.fr/index.php/s/wxCztjYBbQ6zwd6)**<br>All the needed datasets<br>(1.2 Go)|

Have a look about **[How to get and install](https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/wikis/Install-Fidle)** these notebooks and datasets.


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
- **[GTSRB7](GTSRB/07-Show-report.ipynb)** - [Batch reportss](GTSRB/07-Show-report.ipynb)  
Episode 7 : Displaying our jobs report, and the winner is...
- **[GTSRB10](GTSRB/batch_oar.sh)** - [OAR batch script submission](GTSRB/batch_oar.sh)  
Bash script for an OAR batch submission of an ipython code
- **[GTSRB11](GTSRB/batch_slurm.sh)** - [SLURM batch script](GTSRB/batch_slurm.sh)  
Bash script for a Slurm batch submission of an ipython code

### Sentiment analysis with word embedding
- **[IMDB1](IMDB/01-Embedding-Keras.ipynb)** - [Sentiment alalysis with text embedding](IMDB/01-Embedding-Keras.ipynb)  
A very classical example of word embedding with a dataset from Internet Movie Database (IMDB)
- **[IMDB2](IMDB/02-Prediction.ipynb)** - [Reload and reuse a saved model](IMDB/02-Prediction.ipynb)  
Retrieving a saved model to perform a sentiment analysis (movie review)
- **[IMDB3](IMDB/03-LSTM-Keras.ipynb)** - [Sentiment analysis with a LSTM network](IMDB/03-LSTM-Keras.ipynb)  
Still the same problem, but with a network combining embedding and LSTM

### Time series with Recurrent Neural Network (RNN)
- **[SYNOP1](SYNOP/01-Preparation-of-data.ipynb)** - [Time series with RNN - Preparation of data](SYNOP/01-Preparation-of-data.ipynb)  
Episode 1 : Data analysis and creation of a usable dataset
- **[SYNOP2](SYNOP/02-First-predictions.ipynb)** - [Time series with RNN - Try a prediction](SYNOP/02-First-predictions.ipynb)  
Episode 2 : Training session and first predictions
- **[SYNOP3](SYNOP/03-12h-predictions.ipynb)** - [Time series with RNN - 12h predictions](SYNOP/03-12h-predictions.ipynb)  
Episode 3: Attempt to predict in the longer term 

### Unsupervised learning with an autoencoder neural network (AE)
- **[AE1](AE/01-AE-with-MNIST.ipynb)** - [AutoEncoder (AE) with MNIST](AE/01-AE-with-MNIST.ipynb)  
Episode 1 : Model construction and Training
- **[AE2](AE/02-AE-with-MNIST-post.ipynb)** - [AutoEncoder (AE) with MNIST - Analysis](AE/02-AE-with-MNIST-post.ipynb)  
Episode 2 : Exploring our denoiser

### Generative network with Variational Autoencoder (VAE)
- **[VAE1](VAE/01-VAE-with-MNIST.ipynb)** - [Variational AutoEncoder (VAE) with MNIST](VAE/01-VAE-with-MNIST.ipynb)  
Building a simple model with the MNIST dataset
- **[VAE2](VAE/02-VAE-with-MNIST-post.ipynb)** - [Variational AutoEncoder (VAE) with MNIST - Analysis](VAE/02-VAE-with-MNIST-post.ipynb)  
Visualization and analysis of latent space
- **[VAE3](VAE/05-About-CelebA.ipynb)** - [About the CelebA dataset](VAE/05-About-CelebA.ipynb)  
Presentation of the CelebA dataset and problems related to its size
- **[VAE6](VAE/06-Prepare-CelebA-datasets.ipynb)** - [Preparation of the CelebA dataset](VAE/06-Prepare-CelebA-datasets.ipynb)  
Preparation of a clustered dataset, batchable
- **[VAE7](VAE/07-Check-CelebA.ipynb)** - [Checking the clustered CelebA dataset](VAE/07-Check-CelebA.ipynb)  
Check the clustered dataset
- **[VAE8](VAE/08-VAE-with-CelebA==1090048==.ipynb)** - [Variational AutoEncoder (VAE) with CelebA (small)](VAE/08-VAE-with-CelebA==1090048==.ipynb)  
Variational AutoEncoder (VAE) with CelebA (small res. 128x128)
- **[VAE9](VAE/09-VAE-withCelebA-post.ipynb)** - [Variational AutoEncoder (VAE) with CelebA - Analysis](VAE/09-VAE-withCelebA-post.ipynb)  
Exploring latent space of our trained models
- **[VAE10](VAE/batch_slurm.sh)** - [SLURM batch script](VAE/batch_slurm.sh)  
Bash script for SLURM batch submission of VAE notebooks 

### Miscellaneous
- **[ACTF1](Misc/Activation-Functions.ipynb)** - [Activation functions](Misc/Activation-Functions.ipynb)  
Some activation functions, with their derivatives.
- **[NP1](Misc/Numpy.ipynb)** - [A short introduction to Numpy](Misc/Numpy.ipynb)  
Numpy is an essential tool for the Scientific Python.
- **[TSB1](Misc/Using-Tensorboard.ipynb)** - [Tensorboard with/from Jupyter ](Misc/Using-Tensorboard.ipynb)  
4 ways to use Tensorboard from the Jupyter environment
<!-- INDEX_END -->


## Installation

A procedure for **configuring** and **starting Jupyter** is available in the **[Wiki](https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/wikis/Install-Fidle)**.

## Licence

[<img width="100px" src="fidle/img/00-fidle-CC BY-NC-SA.svg"></img>](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
\[en\] Attribution - NonCommercial - ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
\[Fr\] Attribution - Pas d’Utilisation Commerciale - Partage dans les Mêmes Conditions 4.0 International  
See [License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
See [Disclaimer](https://creativecommons.org/licenses/by-nc-sa/4.0/#).  


----
[<img width="80px" src="fidle/img/00-Fidle-logo-01.svg"></img>](#top)
