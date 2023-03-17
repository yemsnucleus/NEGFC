import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv3d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from detection 	import get_intersting_coords

###### NETWORK ##########################
class CNNET(Module):   
	def __init__(self):
		super(CNNET, self).__init__()

		self.conv1 = Conv3d(90, 40, (5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
		self.relu = ReLU(inplace=True)
		self.conv2=Conv3d(40,1,(5,5,5), stride=(1,1,1), padding=2)

	# Defining the forward pass    
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		return x
##########################################

def custom_loss(i, o):
    res = torch.subtract(i,o)
    res = torch.sum(torch.square(res))
    res = torch.mean(res)
    return res

def train_network(cube, table, psf_norm, fwhm=4, bkg_sigma=5, plot=False):

    model = CNNET()
    optimizer = Adam(model.parameters(), lr=0.05)
        
    print(model)

    table_train = torch.from_numpy(table[['x','y','flux']].to_numpy().flatten())
    x_train = torch.from_numpy(cube.reshape(1, cube.shape[0], 1, cube.shape[1], cube.shape[2]))

#######DEFINE TRAINING##################
    def train(epoch):
        model.train()
        tr_loss = 0

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
    
        # prediction for training and validation set
        output_train = model(x_train)
        #prediction = output_train.clone().detach().numpy().reshape(cube.shape[1], cube.shape[2])
        #print(output_train.shape)
        #output_val = model(x_val)

        # computing the training and validation loss
        cnn_table = get_intersting_coords(torch.reshape(output_train, (cube.shape[1],cube.shape[2])), psf_norm, fwhm=fwhm, bkg_sigma=5, plot=False)
        cnn_table.append([{'x':0, 'y':0, 'flux':0, 'fwhm':0, 'snr':0}]*(len(table) - len(cnn_table)), ignore_index=True)
        cnn_table = torch.from_numpy(cnn_table[['x','y','flux']].to_numpy().flatten())
        loss_train = custom_loss(cnn_table, table_train)
        #loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train)
        #val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        #if epoch%2 == 0:
        #    # printing the validation loss
        #    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
########################################################

    ####### TRAINING #############################  
    # defining the number of epochs
    n_epochs = 25
    # empty list to store training losses
    train_losses = []
    for epoch in range(n_epochs):
        train(epoch)

    plt.plot(train_losses)
    plt.show()
    return model(x_train.numpy().reshape(cube.shape))
