import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class convergence_history_CrossEntropyLoss:
  def __init__(self):
    """
    Class to save the training converge properties
    """
    self.loss=nn.CrossEntropyLoss()
    self.history={}                #Save convergence measures in the end of each epoch
    self.history['loss']=[]        #value of the cost function on training data
    self.history['accuracy']=[]         #percentage of  correctly classified instances on training data (if classification)
    self.history['val_loss']=[]    #value of the cost function on validation data
    self.history['val_accuracy']=[]     #percentage of  correctly classified instances on validation data (if classification)
  
  def update(self,current_model,xtrain,ytrain,xtest,ytest):
    
    #convergence information on the training data 
    nb_training_obs=xtrain.shape[0]
    if nb_training_obs>xtest.shape[0]:
        nb_training_obs=xtest.shape[0]
        
    epoch_shuffler=np.arange(xtrain.shape[0]) 
    np.random.shuffle(epoch_shuffler)
    mini_batch_observations = epoch_shuffler[:nb_training_obs]
    var_X_batch = Variable(xtrain[mini_batch_observations,:]).float()
    var_y_batch = Variable(ytrain[mini_batch_observations])
    y_pred_batch = current_model(var_X_batch)
    curr_loss = self.loss(y_pred_batch, var_y_batch)

    self.history['loss'].append(curr_loss.item())
    self.history['accuracy'].append( float( (torch.argmax(y_pred_batch, dim= 1) == var_y_batch).float().mean()) )
    
    #convergence information on the test data 
    var_X_batch = Variable(xtest[:,:]).float()
    var_y_batch = Variable(ytest[:])
    y_pred_batch = current_model(var_X_batch)
    curr_loss = self.loss(y_pred_batch, var_y_batch)

    self.history['val_loss'].append(curr_loss.item())
    self.history['val_accuracy'].append( float( (torch.argmax(y_pred_batch, dim= 1) == var_y_batch).float().mean()) )



class convergence_history_MSELoss:
  def __init__(self):
    """
    Class to save the training converge properties
    """
    self.loss = nn.MSELoss()
    self.MAE_loss = nn.L1Loss()
    self.history={}                #Save convergence measures in the end of each epoch
    self.history['loss']=[]        #value of the cost function on training data
    self.history['mae']=[]         #mean absolute error on training data
    self.history['val_loss']=[]    #value of the cost function on validation data
    self.history['val_mae']=[]     #mean absolute error on validation data
  
  def update(self,current_model,xtrain,ytrain,xtest,ytest):
    
    #convergence information on the training data 
    nb_training_obs=xtrain.shape[0]
    if nb_training_obs>xtest.shape[0]:
        nb_training_obs=xtest.shape[0]
        
    epoch_shuffler=np.arange(xtrain.shape[0]) 
    np.random.shuffle(epoch_shuffler)
    mini_batch_observations = epoch_shuffler[:nb_training_obs]
    var_X_batch = Variable(xtrain[mini_batch_observations,:]).float()
    var_y_batch = Variable(ytrain[mini_batch_observations]).float()
    y_pred_batch = current_model(var_X_batch)
    curr_loss = self.loss(y_pred_batch.view(-1), var_y_batch.view(-1))

    self.history['loss'].append(curr_loss.item())
    self.history['mae'].append(self.MAE_loss(y_pred_batch.view(-1), var_y_batch.view(-1)).item())
    
    #convergence information on the test data 
    var_X_batch = Variable(xtest[:,:]).float()
    var_y_batch = Variable(ytest[:]).float()
    y_pred_batch = current_model(var_X_batch)
    curr_loss = self.loss(y_pred_batch.view(-1), var_y_batch.view(-1))

    self.history['val_loss'].append(curr_loss.item())
    self.history['val_mae'].append(self.MAE_loss(y_pred_batch.view(-1), var_y_batch.view(-1)).item())
