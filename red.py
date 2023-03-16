import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
%matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


def train_network(cube, table,psf_norm, fwhm=4, bkg_sigma=5, plot=opt.plot):

###### NETWORK ##########################
	class CNNET(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        return np.nanmedian(x, axis=0)

    model = CNNET()
    optimizer = Adam(model.parameters(), lr=0.05)
    criterion = MSELoss()
    
    print(model)
##########################################

table_train = table[['x','y','flux']].to_numpy()
x_train = cube


#######DEFINE TRAINING##################
    def train(epoch):
        model.train()
        tr_loss = 0

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
    
        # prediction for training and validation set
        output_train = model(x_train)
        #output_val = model(x_val)

        # computing the training and validation loss
        cnn_table = get_intersting_coords(output_train, psf_norm, fwhm=fwhm, bkg_sigma=5, plot=opt.plot)
        cnn_table.append([{'x':0, 'y':0, 'flux':0, 'fwhm':0, 'snr':0}]*(len(table) - len(cnn_table)), ignore_index=True)
        loss_train = criterion(cnn_table[['x','y','flux']].to_numpy(), table_train)
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

plt.plot(train_losses,
plt.show()

