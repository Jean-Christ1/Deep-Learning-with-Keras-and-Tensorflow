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
Current Version : 0.5.7 DEV  


## Course materials
**[<img width="50px" src="fidle/img/00-Fidle-pdf.svg"></img>
Get the course slides](https://cloud.univ-grenoble-alpes.fr/index.php/s/wxCztjYBbQ6zwd6)**  

[How to get and install](https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/wikis/Install-Fidle) notebooks and datasets  
Some other useful informations are also available in the [wiki](https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/wikis/home)


## Jupyter notebooks

<!-- DO NOT REMOVE THIS TAG !!! -->
<!-- INDEX -->
<!-- INDEX_BEGIN -->
[[LINR1] - Linear regression with direct resolution](LinearReg/01-Linear-Regression.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Direct determination of linear regression   
[[GRAD1] - Linear regression with gradient descent](LinearReg/02-Gradient-descent.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;An example of gradient descent in the simple case of a linear regression.  
[[POLR1] - Complexity Syndrome](LinearReg/03-Polynomial-Regression.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Illustration of the problem of complexity with the polynomial regression  
[[LOGR1] - Logistic regression, in pure Tensorflow](LinearReg/04-Logistic-Regression.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Logistic Regression with Mini-Batch Gradient Descent using pure TensorFlow.   
[[PER57] - Perceptron Model 1957](IRIS/01-Simple-Perceptron.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A simple perceptron, with the IRIS dataset.  
[[BHP1] - Regression with a Dense Network (DNN)](BHPD/01-DNN-Regression.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A Simple regression with a Dense Neural Network (DNN) - BHPD dataset  
[[BHP2] - Regression with a Dense Network (DNN) - Advanced code](BHPD/02-DNN-Regression-Premium.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;More advanced example of DNN network code - BHPD dataset  
[[MNIST1] - Simple classification with DNN](MNIST/01-DNN-MNIST.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example of classification with a fully connected neural network  
[[GTS1] - CNN with GTSRB dataset - Data analysis and preparation](GTSRB/01-Preparation-of-data.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 1 : Data analysis and creation of a usable dataset  
[[GTS2] - CNN with GTSRB dataset - First convolutions](GTSRB/02-First-convolutions.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 2 : First convolutions and first results  
[[GTS3] - CNN with GTSRB dataset - Monitoring ](GTSRB/03-Tracking-and-visualizing.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 3 : Monitoring and analysing training, managing checkpoints  
[[GTS4] - CNN with GTSRB dataset - Data augmentation ](GTSRB/04-Data-augmentation.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 4 : Improving the results with data augmentation  
[[GTS5] - CNN with GTSRB dataset - Full convolutions ](GTSRB/05-Full-convolutions.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 5 : A lot of models, a lot of datasets and a lot of results.  
[[GTS6] - CNN with GTSRB dataset - Full convolutions as a batch](GTSRB/06-Notebook-as-a-batch.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 6 : Run Full convolution notebook as a batch  
[[GTS7] - CNN with GTSRB dataset - Show reports](GTSRB/07-Show-report.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 7 : Displaying the reports of the different jobs  
[[TSB1] - Tensorboard with/from Jupyter ](GTSRB/99-Scripts-Tensorboard.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4 ways to use Tensorboard from the Jupyter environment  
[[BASH1] - OAR batch script](GTSRB/batch_oar.sh)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bash script for OAR batch submission of GTSRB notebook  
[[BASH2] - SLURM batch script](GTSRB/batch_slurm.sh)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bash script for SLURM batch submission of GTSRB notebooks  
[[IMDB1] - Text embedding with IMDB](IMDB/01-Embedding-Keras.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A very classical example of word embedding for text classification (sentiment analysis)  
[[IMDB2] - Text embedding with IMDB - Reloaded](IMDB/02-Prediction.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example of reusing a previously saved model  
[[IMDB3] - Text embedding/LSTM model with IMDB](IMDB/03-LSTM-Keras.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Still the same problem, but with a network combining embedding and LSTM  
[[SYNOP1] - Time series with RNN - Preparation of data](SYNOP/01-Preparation-of-data.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 1 : Data analysis and creation of a usable dataset  
[[SYNOP2] - Time series with RNN - Try a prediction](SYNOP/02-First-predictions.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 2 : Training session and first predictions  
[[SYNOP3] - Time series with RNN - 12h predictions](SYNOP/03-12h-predictions.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 3: Attempt to predict in the longer term   
[[VAE1] - Variational AutoEncoder (VAE) with MNIST](VAE/01-VAE-with-MNIST.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 1 : Model construction and Training  
[[VAE1] - Variational AutoEncoder (VAE) with MNIST](VAE/01-VAE-with-MNIST.nbconvert.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 1 : Model construction and Training  
[[VAE2] - Variational AutoEncoder (VAE) with MNIST - Analysis](VAE/02-VAE-with-MNIST-post.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 2 : Exploring our latent space  
[[VAE3] - About the CelebA dataset](VAE/03-About-CelebA.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 3 : About the CelebA dataset, a more fun dataset ;-)  
[[VAE4] - Preparation of the CelebA dataset](VAE/04-Prepare-CelebA-datasets.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 4 : Preparation of a clustered dataset, batchable  
[[VAE5] - Checking the clustered CelebA dataset](VAE/05-Check-CelebA.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 5 :\tChecking the clustered dataset  
[[VAE6] - Variational AutoEncoder (VAE) with CelebA (small)](VAE/06-VAE-with-CelebA-s.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 6 : Variational AutoEncoder (VAE) with CelebA (small res.)  
[[VAE7] - Variational AutoEncoder (VAE) with CelebA (medium)](VAE/07-VAE-with-CelebA-m.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 7 : Variational AutoEncoder (VAE) with CelebA (medium res.)  
[[VAE8] - Variational AutoEncoder (VAE) with CelebA - Analysis](VAE/08-VAE-withCelebA-post.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Episode 8 : Exploring latent space of our trained models  
[[BASH1] - OAR batch script](VAE/batch_oar.sh)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bash script for OAR batch submission of VAE notebook  
[[BASH2] - SLURM batch script](VAE/batch_slurm.sh)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bash script for SLURM batch submission of VAE notebooks  
[[ACTF1] - Activation functions](Misc/Activation-Functions.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Some activation functions, with their derivatives.  
[[NP1] - A short introduction to Numpy](Misc/Numpy.ipynb)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Numpy is an essential tool for the Scientific Python.  
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
