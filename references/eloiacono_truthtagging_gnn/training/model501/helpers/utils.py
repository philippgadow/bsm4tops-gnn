import json
import logging
import os
import shutil
from pathlib import Path

import torch


def save_checkpoint(state_dict, is_best, path, epoch=None):

	"""Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
	checkpoint + 'best.pth.tar'
	
	Args:
		state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
		is_best: (bool) True if it is the best model seen till now
		path: (string) folder where parameters are to be saved
		epoch: epoch number, so that we can save the model for each epoch
	"""

	filepath = os.path.join(path, 'last_checkpoint.pth.tar')

	if not os.path.exists(path):
		print("Checkpoint Directory does not exist! Making directory {}".format(filepath))
		os.mkdir(path)
	else:
		if not os.path.isfile(filepath):
			print("Checkpoint doesn't exist. Will be created by torch")
	

	torch.save(state_dict, filepath)

	if epoch != None:
		epoch_path = os.path.join(path, 'epoch' + str(epoch))
		Path(epoch_path).mkdir(parents=True, exist_ok=True)
		shutil.copyfile(filepath, os.path.join(epoch_path, 'checkpoint.' + str(epoch) + '.pth.tar'))

	if is_best==True:
		shutil.copyfile(filepath, os.path.join(path, 'best_checkpoint.pth.tar'))




def load_checkpoint(checkpoint_file, path, model, optimizer=None):

	"""Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
	optimizer assuming it is present in checkpoint.
	Args:
		checkpoint: (string) filename which needs to be loaded
		model: (torch.nn.Module) model for which the parameters are loaded
		optimizer: (torch.optim) optional: resume optimizer from checkpoint
	"""
	filepath = os.path.join(path, checkpoint_file + '.pth.tar')
	print("Restoring parameters from {}".format(filepath))

	if not os.path.exists(filepath):
		print("Checkpoint file not found!")
		print("Training network from scratch...")

		return

	try:
		checkpoint = torch.load(filepath)
	except RuntimeError:
		checkpoint = torch.load(filepath, map_location=torch.device('cpu'))		

	model.load_state_dict(checkpoint['model_dict'])

	if optimizer is not None:
		optimizer.load_state_dict(checkpoint['optim_dict'])


def save_metrics(metric_dict, path):
	
	""" Save the metrics (in a dict form) in the given file path. If the file is not available, it'll create one.
	Args:
		path:  path to the directory where json file will be stored
		metric_dict: dictionary containing the metrics to be stored
	"""

	filepath = os.path.join(path, 'metrics.json')
	
	if not os.path.exists(path):
		print("Metrics Directory does not exist! Making directory {}".format(filepath))
		os.mkdir(path)
	else:
		if not os.path.isfile(filepath):
			print("Metrics file doesn't exist. Creating file")

	with open(filepath, 'w') as f:
		json.dump(metric_dict, f)




def load_metrics(path, metric_dict):

	""" Loads the metrics (in a dict form) form the given file path (to the metric.json file)
	Args:
		path: path to the directory where the metrics.json file is
	"""

	filepath = os.path.join(path, 'metrics.json')	
	print("Restoring metrics from {}".format(filepath))

	if not os.path.exists(filepath):
		print("Metrics file not found!")
		print("Metrics will be stored assuming this to be the first epoch")
		return

	filepath = os.path.join(path, 'metrics.json')	
	with open(filepath, 'r') as f:
		data = json.load(f)

		for key in data.keys():
			metric_dict[key] = data[key]




def save_model_info(path, net, epochs, lr, milestones, gamma, batch_size):

	'''
	Writes the model info into a json file
	Args:
		param_dict
	'''

	filepath = Path(os.path.join(path, 'about.json'))

	if filepath.is_file():
		with open(filepath, 'r') as f:
			about_dict = json.load(f)
	else:
		about_dict = {
			'net'    	 : None,
			'epochs' 	 : 0,
			'lr'		 : 1,
			'milestones' : [],
			'gamma'  	 : 1,
			'batch_size' : 1,
		}

	# update
	with open(filepath, 'w+') as f:
		about_dict['net'] = net
		about_dict['epochs'] += epochs
		about_dict['lr'] = lr
		about_dict['milestones'].extend(milestones)
		about_dict['gamma'] = gamma
		about_dict['batch_size'] = batch_size

		json.dump(about_dict, f)

