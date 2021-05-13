#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import os 
print(tf.__version__)

dir_path = os.getcwd()
dirName = 'SGD'

try:
   os.mkdir(dirName)
   print("Directory ", dirName , "was created")
except FileExistsError:
  print("Directory ", dirName , "already exists")



  
  
 
  

  

path_to_save = dir_path + '/' + dirName
lr_range=[0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
momentum_range = [0.95, 0.90, 0.85]
nesterov_range = ['True', 'False']
decay_val = 1e-6


for nesterov_val in nesterov_range:
	for lr_val in lr_range:
		for momentum_val in momentum_range:
			dataset = pd.read_csv("US_TN_season_1_proc.csv")	
			dataset.tail()

			#Check for NaN in this table and drop them if there are
			dataset.isna().sum()
			dataset.dropna()

			#Split the dataset into a training set and a test set.
			dataset_orig = dataset #keep a backup of the original dataset. Might be useful.

			# Remove extra variables from the dataset (keep just the 47 predictors and the 'bcmean' (what we are predicting)
			dataset = dataset.drop(["RVunif_bc","mean","std","cv","longitude","latitude","RVunif"],axis=1)
			dataset_orig2 = dataset
			dataset.to_csv('dataset_clean.csv',index=False) #Saving the csv file just for easier visualization of the raw data

			# Rescale: differences in scales accross input variables may increase the difficulty of the problem being modeled and results on unstable weights for connections
			sc = MinMaxScaler(feature_range = (0,1)) #Scaling features to a range between 0 and 1

			# Scaling and translating each feature to our chosen range
			dataset = sc.fit_transform(dataset) 
			dataset = pd.DataFrame(dataset, columns = dataset_orig2.columns)
			dataset_scaled = dataset #Just backup
			inverse_data = sc.inverse_transform(dataset) #just to make sure it works

			train_dataset = dataset.sample(frac=0.8,random_state=0)
			test_dataset = dataset.drop(train_dataset.index)
			train_dataset_orig = dataset_orig2.sample(frac=0.8,random_state=0) #just backup
			test_dataset_orig =  dataset_orig2.drop(train_dataset_orig.index) #just backup

			#Inspect the original mean (still missing some formatting)
			sns.set()
			f, (ax1,ax2) = plt.subplots(2, 1,sharex=True)
			sns.distplot(train_dataset["bcmean"],hist=True,kde=False,bins=75,color='darkblue',  ax=ax1, axlabel=False)
			sns.kdeplot(train_dataset["bcmean"],bw=0.15,legend=True,color='darkblue', ax=ax2)

			ax1.set_title('Original  histogram')
			ax1.legend(['bcmean'])
			ax2.set_title('KDE')
			ax2.set_xlabel('Mean Concentration N')
			ax1.set_ylabel('Count')
			ax2.set_ylabel('Dist')
			# plt.show()

			#Check the overall stats
			train_stats = train_dataset.describe()
			train_stats.pop('bcmean') #because that is what we are trying to predict
			train_stats = train_stats.transpose()
			train_stats #now train_stats has 47 predictors (as described in the paper).

			# Remove the output from our list of predictors
			train_dataset.to_csv('train_dataset.csv',index=False) 
			train_labels = train_dataset.pop('bcmean')
			test_dataset.to_csv('test_dataset.csv',index=False)
			test_labels = test_dataset.pop('bcmean')


			# Inspect the joint distribution of a few pairs of columns from the training set
			# We can observe that the process of scalling the data did not affect the skewness of the data
			#sns.pairplot(train_dataset[["lc09", "lc07", "hydro05", "hydro07","soil01","dem"]], diag_kind="kde")
			## plt.show()

			#Normalize the data because the data has very different ranges (not really crucial for the prediction to use the raw data)
			#def norm(x):
			# return (x - train_stats['mean']) / train_stats['std']
			normed_train_data = train_dataset #norm(train_dataset)
			normed_test_data = test_dataset #norm(test_dataset)

			#Train the model for 1000 epochs
			# Display training progress by printing a single dot for each completed epoch
			class PrintDot(keras.callbacks.Callback):
			  def on_epoch_end(self, epoch, logs):
			    if epoch % 100 == 0: print('')
			    print('.', end='')

			# Build the model. 'Sequential' model with two densely connected hidden layers,and an output layer that returns a single, continous value
			# The architecture can be freely modified (pretty much whatever works better for your data)
			def build_model():
			  model = keras.Sequential([
			    layers.Dense(64,kernel_initializer='normal',activation=tf.nn.relu,input_shape=[len(train_dataset.keys())]),
			    layers.Dropout(0.2),
			    layers.Dense(64, activation=tf.nn.relu),
			    layers.Dropout(0.2),
			    layers.Dense(1,kernel_initializer='normal',activation='relu')
			  ])
			  optimizer = tf.keras.optimizers.SGD(lr=lr_val, momentum=momentum_val, decay=decay_val, nesterov=nesterov_val) #many other options for optmizer
			  model.compile(loss='mean_squared_error',
			                optimizer=optimizer,
			                metrics=['mean_absolute_error', 'mean_squared_error']) #When dealing with classification, 'accuracy' is very useful as well
			  return model

			model = build_model()
			model.summary()

			#Take 10 samples from training dataset for quick test
			example_batch = normed_train_data[:10]
			example_result = model.predict(example_batch)
			example_result

			# The patience parameter is the amount of epochs to check for improvement
			early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

			EPOCHS = 5000

			history = model.fit(
			  normed_train_data, train_labels,
			  epochs=EPOCHS, 
			  validation_split=0.1,
			  shuffle=True, verbose=2,
			  callbacks=[early_stop, PrintDot()])

			#Plot the progress of the training
			hist = pd.DataFrame(history.history)
			hist['epoch'] = history.epoch
			hist.tail()

			def plot_history(history):
			  hist = pd.DataFrame(history.history)
			  hist['epoch'] = history.epoch
			  plt.figure()
			  plt.xlabel('Epoch')
			  plt.ylabel('Mean Abs Error [Mean Conc. N]')
			  plt.plot(hist['epoch'], hist['mean_absolute_error'],
			           label='Train Error')
			  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
			           label = 'Val Error')
			  #plt.ylim([0,1])
			  plt.legend()
			  plt.savefig(path_to_save + '/mean_asb_error_lr' + str(lr_val) + '_moment' + str(momentum_val) + '_nest' + str(nesterov_val) + '.png', bbox_inches='tight')
			  plt.figure()
			  plt.xlabel('Epoch')
			  plt.ylabel('Mean Square Error [$(Mean Conc.)^2$]')
			  plt.plot(hist['epoch'], hist['mean_squared_error'],
			           label='Train Error')
			  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
			           label = 'Val Error')
			  #plt.ylim([0,3])
			  plt.legend()
			  plt.savefig(path_to_save + '/mean_sq_error_lr' + str(lr_val) + '_moment' + str(momentum_val) + '_nest' + str(nesterov_val) + '.png', bbox_inches='tight')
			  # plt.show()

			plot_history(history)

			#Time for a real test
			f, (ax1,ax2) = plt.subplots(1,2, sharey=True)
			test_predictions = model.predict(normed_test_data).flatten()
			r = r2_score(test_labels, test_predictions)
			ax1.scatter(test_labels, test_predictions,alpha=0.5, label='$R^2$ = %.3f' % (r))
			ax1.legend(loc="upper left")
			ax1.set_xlabel('True Values [Mean Conc.]')
			ax1.set_ylabel('Predictions [Mean Conc.]')
			ax1.axis('equal')
			ax1.axis('square')
			ax1.set_xlim([0,1])
			ax1.set_ylim([0,1])
			_ = ax1.plot([-100, 100], [-100, 100], 'r:')
			ax1.set_title('Test dataset')
			f.set_figheight(30)
			f.set_figwidth(10)
			#plt.show()
			#plt.close('all')

			#Whole dataset
			dataset_labels = dataset.pop('bcmean')
			normed_dataset = dataset
			dataset_predictions = model.predict(normed_dataset).flatten()
			r = r2_score(dataset_labels, dataset_predictions)
			ax2.scatter(dataset_labels, dataset_predictions, alpha=0.5, label='$R^2$ = %.3f' % (r))
			ax2.legend(loc="upper left")
			ax2.set_xlabel('True Values [Mean Conc.]')
			ax2.set_ylabel('Predictions [Mean Conc.]')
			ax2.axis('equal')
			ax2.axis('square')
			ax2.set_xlim([0,1])
			ax2.set_ylim([0,1])
			_ = ax2.plot([-100, 100], [-100, 100], 'r:')
			ax2.set_title('Whole dataset')
			# plt.show()
			plt.savefig(path_to_save + '/R_scaled_lr' + str(lr_val) + '_moment' + str(momentum_val) + '_nest' + str(nesterov_val) + '.png', bbox_inches='tight')
			#plt.close('all')

			#Undo scale step
			normed_test_data['bcmean'] = test_predictions
			inverse_data = sc.inverse_transform(normed_test_data)
			inverse_data = pd.DataFrame(inverse_data, columns = dataset_orig2.columns)
			test_predictions = inverse_data.pop('bcmean')
			test_labels = test_dataset_orig.pop('bcmean')
			f, (ax1,ax2) = plt.subplots(1,2, sharey=True)
			r = r2_score(test_labels, test_predictions)
			ax1.scatter(test_labels, test_predictions, alpha=0.5, label='$R^2$ = %.3f' % (r))
			ax1.legend(loc="upper left")
			ax1.set_xlabel('True Values [Mean Conc.]')
			ax1.set_ylabel('Predictions [Mean Conc.]')
			ax1.axis('equal')
			ax1.axis('square')
			ax1.set_xlim([-3,3])
			ax1.set_ylim([-3,3])
			_ = ax1.plot([-100, 100], [-100, 100], 'r:')
			ax1.set_title('Test dataset')
			f.set_figheight(30)
			f.set_figwidth(10)
			#plt.show()

			#Whole dataset
			normed_dataset['bcmean'] = dataset_predictions
			inverse_data = sc.inverse_transform(normed_dataset)
			inverse_data = pd.DataFrame(inverse_data, columns = dataset_orig2.columns)
			dataset_predictions = inverse_data.pop('bcmean')
			dataset_labels = dataset_orig2.pop('bcmean')
			r = r2_score(dataset_labels, dataset_predictions)
			ax2.scatter(dataset_labels, dataset_predictions, alpha=0.5, label='$R^2$ = %.3f' % (r))
			ax2.legend(loc="upper left")
			ax2.set_xlabel('True Values [Mean Conc.]')
			ax2.set_ylabel('Predictions [Mean Conc.]')
			ax2.axis('equal')
			ax2.axis('square')
			ax2.set_xlim([-3,3])
			ax2.set_ylim([-3,3])
			_ = ax2.plot([-100, 100], [-100, 100], 'r:')
			ax2.set_title('Whole dataset')
			# plt.show()
			plt.savefig(path_to_save + '/R_unscaled_lr' + str(lr_val) + '_moment' + str(momentum_val) + '_nest' + str(nesterov_val) + '.png', bbox_inches='tight')
			del model
