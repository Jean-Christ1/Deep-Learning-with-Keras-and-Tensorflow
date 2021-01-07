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
| | |
|--|--|
|LINR1| [Linear regression with direct resolution](LinearReg/01-Linear-Regression.ipynb)<br>Direct determination of linear regression |
|GRAD1| [Linear regression with gradient descent](LinearReg/02-Gradient-descent.ipynb)<br>An example of gradient descent in the simple case of a linear regression.|
|POLR1| [Complexity Syndrome](LinearReg/03-Polynomial-Regression.ipynb)<br>Illustration of the problem of complexity with the polynomial regression|
|LOGR1| [Logistic regression, with sklearn](LinearReg/04-Logistic-Regression.ipynb)<br>Logistic Regression using Sklearn|
|PER57| [Perceptron Model 1957](IRIS/01-Simple-Perceptron.ipynb)<br>A simple perceptron, with the IRIS dataset.|
|MNIST1| [Simple classification with DNN](MNIST/01-DNN-MNIST.ipynb)<br>Example of classification with a fully connected neural network|
|GTSRB1| [CNN with GTSRB dataset - Data analysis and preparation](GTSRB/01-Preparation-of-data.ipynb)<br>Episode 1 : Data analysis and creation of a usable dataset|
|GTSRB2| [CNN with GTSRB dataset - First convolutions](GTSRB/02-First-convolutions.ipynb)<br>Episode 2 : First convolutions and first results|
|GTSRB3| [CNN with GTSRB dataset - Monitoring ](GTSRB/03-Tracking-and-visualizing.ipynb)<br>Episode 3 : Monitoring and analysing training, managing checkpoints|
|GTSRB4| [CNN with GTSRB dataset - Data augmentation ](GTSRB/04-Data-augmentation.ipynb)<br>Episode 4 : Improving the results with data augmentation|
|GTSRB5| [CNN with GTSRB dataset - Full convolutions ](GTSRB/05-Full-convolutions.ipynb)<br>Episode 5 : A lot of models, a lot of datasets and a lot of results.|
|GTSRB6| [Full convolutions as a batch](GTSRB/06-Notebook-as-a-batch.ipynb)<br>Episode 6 : Run Full convolution notebook as a batch|
|GTSRB7| [CNN with GTSRB dataset - Show reports](GTSRB/07-Show-report.ipynb)<br>Episode 7 : Displaying a jobs report|
|GTSRB10| [OAR batch submission](GTSRB/batch_oar.sh)<br>Bash script for OAR batch submission of GTSRB notebook |
|GTSRB11| [SLURM batch script](GTSRB/batch_slurm.sh)<br>Bash script for SLURM batch submission of GTSRB notebooks |
|IMDB1| [Text embedding with IMDB](IMDB/01-Embedding-Keras.ipynb)<br>A very classical example of word embedding for text classification (sentiment analysis)|
|IMDB2| [Text embedding with IMDB - Reloaded](IMDB/02-Prediction.ipynb)<br>Example of reusing a previously saved model|
|IMDB3| [Text embedding/LSTM model with IMDB](IMDB/03-LSTM-Keras.ipynb)<br>Still the same problem, but with a network combining embedding and LSTM|
|SYNOP1| [Time series with RNN - Preparation of data](SYNOP/01-Preparation-of-data.ipynb)<br>Episode 1 : Data analysis and creation of a usable dataset|
|SYNOP2| [Time series with RNN - Try a prediction](SYNOP/02-First-predictions.ipynb)<br>Episode 2 : Training session and first predictions|
|SYNOP3| [Time series with RNN - 12h predictions](SYNOP/03-12h-predictions.ipynb)<br>Episode 3: Attempt to predict in the longer term |
|VAE1| [Variational AutoEncoder (VAE) with MNIST](VAE/01-VAE-with-MNIST.ipynb)<br>Building a simple model with the MNIST dataset|
|VAE2| [Variational AutoEncoder (VAE) with MNIST - Analysis](VAE/02-VAE-with-MNIST-post.ipynb)<br>Visualization and analysis of latent space|
|VAE3| [About the CelebA dataset](VAE/05-About-CelebA.ipynb)<br>Presentation of the CelebA dataset and problems related to its size|
|VAE6| [Preparation of the CelebA dataset](VAE/06-Prepare-CelebA-datasets.ipynb)<br>Preparation of a clustered dataset, batchable|
|VAE7| [Checking the clustered CelebA dataset](VAE/07-Check-CelebA.ipynb)<br>Check the clustered dataset|
|VAE8| [Variational AutoEncoder (VAE) with CelebA](VAE/08-VAE-with-CelebA.ipynb)<br>Building a VAE and train it, using a data generator|
|VAE9| [Variational AutoEncoder (VAE) with CelebA - Analysis](VAE/09-VAE-withCelebA-post.ipynb)<br>Exploring latent space of our trained models|
|VAE10| [SLURM batch script](VAE/batch_slurm.sh)<br>Bash script for SLURM batch submission of VAE notebooks |
|ACTF1| [Activation functions](Misc/Activation-Functions.ipynb)<br>Some activation functions, with their derivatives.|
|NP1| [A short introduction to Numpy](Misc/Numpy.ipynb)<br>Numpy is an essential tool for the Scientific Python.|
|TSB1| [Tensorboard with/from Jupyter ](Misc/Using-Tensorboard.ipynb)<br>4 ways to use Tensorboard from the Jupyter environment|
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
