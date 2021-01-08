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
<style  type="text/css">

.fid_line{
    padding-top: 10px
}

.fid_id {    
    font-size:1.em;
    color:black;
    font-weight: bold; 
    padding:0px;
    margin-left: 20px;
    display: inline-block;
    width: 60px;
    }

.fid_desc {    
    font-size:1.em;
    padding:0px;
    margin-left: 85px;
    display: inline-block;
    width: 600px;
    }



div.fid_section {    
    font-size:1.2em;
    color:black;
    margin-left: 0px;
    margin-top: 12px;
    margin-bottom:8px;
    border-bottom: solid;
    border-block-width: 1px;
    border-block-color: #dadada;
    width: 700px;
    }

</style>
<div class="fid_section">Linear and logistic regression</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="LinearReg/01-Linear-Regression.ipynb">LINR1</a>
                     </span> <a href="LinearReg/01-Linear-Regression.ipynb">Linear regression with direct resolution</a><br>
                     <span class="fid_desc">Direct determination of linear regression </span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="LinearReg/02-Gradient-descent.ipynb">GRAD1</a>
                     </span> <a href="LinearReg/02-Gradient-descent.ipynb">Linear regression with gradient descent</a><br>
                     <span class="fid_desc">An example of gradient descent in the simple case of a linear regression.</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="LinearReg/03-Polynomial-Regression.ipynb">POLR1</a>
                     </span> <a href="LinearReg/03-Polynomial-Regression.ipynb">Complexity Syndrome</a><br>
                     <span class="fid_desc">Illustration of the problem of complexity with the polynomial regression</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="LinearReg/04-Logistic-Regression.ipynb">LOGR1</a>
                     </span> <a href="LinearReg/04-Logistic-Regression.ipynb">Logistic regression, with sklearn</a><br>
                     <span class="fid_desc">Logistic Regression using Sklearn</span>
                 </div>
        
<div class="fid_section">Perceptron Model 1957</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="IRIS/01-Simple-Perceptron.ipynb">PER57</a>
                     </span> <a href="IRIS/01-Simple-Perceptron.ipynb">Perceptron Model 1957</a><br>
                     <span class="fid_desc">A simple perceptron, with the IRIS dataset.</span>
                 </div>
        
<div class="fid_section">Basic regression using DNN</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="BHPD/01-DNN-Regression.ipynb">BHPD1</a>
                     </span> <a href="BHPD/01-DNN-Regression.ipynb">Regression with a Dense Network (DNN)</a><br>
                     <span class="fid_desc">A Simple regression with a Dense Neural Network (DNN) - BHPD dataset</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="BHPD/02-DNN-Regression-Premium.ipynb">BHPD2</a>
                     </span> <a href="BHPD/02-DNN-Regression-Premium.ipynb">Regression with a Dense Network (DNN) - Advanced code</a><br>
                     <span class="fid_desc">More advanced example of DNN network code - BHPD dataset</span>
                 </div>
        
<div class="fid_section">Basic classification using a DNN</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="MNIST/01-DNN-MNIST.ipynb">MNIST1</a>
                     </span> <a href="MNIST/01-DNN-MNIST.ipynb">Simple classification with DNN</a><br>
                     <span class="fid_desc">Example of classification with a fully connected neural network</span>
                 </div>
        
<div class="fid_section">Images classification with Convolutional Neural Networks (CNN)</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/01-Preparation-of-data.ipynb">GTSRB1</a>
                     </span> <a href="GTSRB/01-Preparation-of-data.ipynb">CNN with GTSRB dataset - Data analysis and preparation</a><br>
                     <span class="fid_desc">Episode 1 : Data analysis and creation of a usable dataset</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/02-First-convolutions.ipynb">GTSRB2</a>
                     </span> <a href="GTSRB/02-First-convolutions.ipynb">CNN with GTSRB dataset - First convolutions</a><br>
                     <span class="fid_desc">Episode 2 : First convolutions and first results</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/03-Tracking-and-visualizing.ipynb">GTSRB3</a>
                     </span> <a href="GTSRB/03-Tracking-and-visualizing.ipynb">CNN with GTSRB dataset - Monitoring </a><br>
                     <span class="fid_desc">Episode 3 : Monitoring and analysing training, managing checkpoints</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/04-Data-augmentation.ipynb">GTSRB4</a>
                     </span> <a href="GTSRB/04-Data-augmentation.ipynb">CNN with GTSRB dataset - Data augmentation </a><br>
                     <span class="fid_desc">Episode 4 : Improving the results with data augmentation</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/05-Full-convolutions.ipynb">GTSRB5</a>
                     </span> <a href="GTSRB/05-Full-convolutions.ipynb">CNN with GTSRB dataset - Full convolutions </a><br>
                     <span class="fid_desc">Episode 5 : A lot of models, a lot of datasets and a lot of results.</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/06-Notebook-as-a-batch.ipynb">GTSRB6</a>
                     </span> <a href="GTSRB/06-Notebook-as-a-batch.ipynb">Full convolutions as a batch</a><br>
                     <span class="fid_desc">Episode 6 : Run Full convolution notebook as a batch</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/07-Show-report.ipynb">GTSRB7</a>
                     </span> <a href="GTSRB/07-Show-report.ipynb">CNN with GTSRB dataset - Show reports</a><br>
                     <span class="fid_desc">Episode 7 : Displaying a jobs report</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/batch_oar.sh">GTSRB10</a>
                     </span> <a href="GTSRB/batch_oar.sh">OAR batch submission</a><br>
                     <span class="fid_desc">Bash script for OAR batch submission of GTSRB notebook </span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="GTSRB/batch_slurm.sh">GTSRB11</a>
                     </span> <a href="GTSRB/batch_slurm.sh">SLURM batch script</a><br>
                     <span class="fid_desc">Bash script for SLURM batch submission of GTSRB notebooks </span>
                 </div>
        
<div class="fid_section">Sentiment analysis with word embedding</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="IMDB/01-Embedding-Keras.ipynb">IMDB1</a>
                     </span> <a href="IMDB/01-Embedding-Keras.ipynb">Text embedding with IMDB</a><br>
                     <span class="fid_desc">A very classical example of word embedding for text classification (sentiment analysis)</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="IMDB/02-Prediction.ipynb">IMDB2</a>
                     </span> <a href="IMDB/02-Prediction.ipynb">Text embedding with IMDB - Reloaded</a><br>
                     <span class="fid_desc">Example of reusing a previously saved model</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="IMDB/03-LSTM-Keras.ipynb">IMDB3</a>
                     </span> <a href="IMDB/03-LSTM-Keras.ipynb">Text embedding/LSTM model with IMDB</a><br>
                     <span class="fid_desc">Still the same problem, but with a network combining embedding and LSTM</span>
                 </div>
        
<div class="fid_section">Time series with Recurrent Neural Network (RNN)</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="SYNOP/01-Preparation-of-data.ipynb">SYNOP1</a>
                     </span> <a href="SYNOP/01-Preparation-of-data.ipynb">Time series with RNN - Preparation of data</a><br>
                     <span class="fid_desc">Episode 1 : Data analysis and creation of a usable dataset</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="SYNOP/02-First-predictions.ipynb">SYNOP2</a>
                     </span> <a href="SYNOP/02-First-predictions.ipynb">Time series with RNN - Try a prediction</a><br>
                     <span class="fid_desc">Episode 2 : Training session and first predictions</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="SYNOP/03-12h-predictions.ipynb">SYNOP3</a>
                     </span> <a href="SYNOP/03-12h-predictions.ipynb">Time series with RNN - 12h predictions</a><br>
                     <span class="fid_desc">Episode 3: Attempt to predict in the longer term </span>
                 </div>
        
<div class="fid_section">Unsupervised learning with an autoencoder neural network (AE)</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="AE/01-AE-with-MNIST.ipynb">AE1</a>
                     </span> <a href="AE/01-AE-with-MNIST.ipynb">AutoEncoder (AE) with MNIST</a><br>
                     <span class="fid_desc">Episode 1 : Model construction and Training</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="AE/02-AE-with-MNIST-post.ipynb">AE2</a>
                     </span> <a href="AE/02-AE-with-MNIST-post.ipynb">AutoEncoder (AE) with MNIST - Analysis</a><br>
                     <span class="fid_desc">Episode 2 : Exploring our denoiser</span>
                 </div>
        
<div class="fid_section">Generative network with Variational Autoencoder (VAE)</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/01-VAE-with-MNIST.ipynb">VAE1</a>
                     </span> <a href="VAE/01-VAE-with-MNIST.ipynb">Variational AutoEncoder (VAE) with MNIST</a><br>
                     <span class="fid_desc">Building a simple model with the MNIST dataset</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/02-VAE-with-MNIST-post.ipynb">VAE2</a>
                     </span> <a href="VAE/02-VAE-with-MNIST-post.ipynb">Variational AutoEncoder (VAE) with MNIST - Analysis</a><br>
                     <span class="fid_desc">Visualization and analysis of latent space</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/05-About-CelebA.ipynb">VAE3</a>
                     </span> <a href="VAE/05-About-CelebA.ipynb">About the CelebA dataset</a><br>
                     <span class="fid_desc">Presentation of the CelebA dataset and problems related to its size</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/06-Prepare-CelebA-datasets.ipynb">VAE6</a>
                     </span> <a href="VAE/06-Prepare-CelebA-datasets.ipynb">Preparation of the CelebA dataset</a><br>
                     <span class="fid_desc">Preparation of a clustered dataset, batchable</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/07-Check-CelebA.ipynb">VAE7</a>
                     </span> <a href="VAE/07-Check-CelebA.ipynb">Checking the clustered CelebA dataset</a><br>
                     <span class="fid_desc">Check the clustered dataset</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/08-VAE-with-CelebA==1090048==.ipynb">VAE8</a>
                     </span> <a href="VAE/08-VAE-with-CelebA==1090048==.ipynb">Variational AutoEncoder (VAE) with CelebA (small)</a><br>
                     <span class="fid_desc">Variational AutoEncoder (VAE) with CelebA (small res. 128x128)</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/09-VAE-withCelebA-post.ipynb">VAE9</a>
                     </span> <a href="VAE/09-VAE-withCelebA-post.ipynb">Variational AutoEncoder (VAE) with CelebA - Analysis</a><br>
                     <span class="fid_desc">Exploring latent space of our trained models</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="VAE/batch_slurm.sh">VAE10</a>
                     </span> <a href="VAE/batch_slurm.sh">SLURM batch script</a><br>
                     <span class="fid_desc">Bash script for SLURM batch submission of VAE notebooks </span>
                 </div>
        
<div class="fid_section">Miscellaneous</div>
<div class="fid_line">
                     <span class="fid_id">
                         <a href="Misc/Activation-Functions.ipynb">ACTF1</a>
                     </span> <a href="Misc/Activation-Functions.ipynb">Activation functions</a><br>
                     <span class="fid_desc">Some activation functions, with their derivatives.</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="Misc/Numpy.ipynb">NP1</a>
                     </span> <a href="Misc/Numpy.ipynb">A short introduction to Numpy</a><br>
                     <span class="fid_desc">Numpy is an essential tool for the Scientific Python.</span>
                 </div>
        
<div class="fid_line">
                     <span class="fid_id">
                         <a href="Misc/Using-Tensorboard.ipynb">TSB1</a>
                     </span> <a href="Misc/Using-Tensorboard.ipynb">Tensorboard with/from Jupyter </a><br>
                     <span class="fid_desc">4 ways to use Tensorboard from the Jupyter environment</span>
                 </div>
        
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
