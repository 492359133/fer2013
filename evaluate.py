# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse, sys
import numpy as np
import datetime
import shutil
from expressionnet import ExpressionNet
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fer import FER2013
from sklearn.metrics import confusion_matrix
import itertools
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--save_dir_1', type = str, help = 'dir to save result ckpt files', default = 'checkpoints_1_resnet80_rsa/')
parser.add_argument('--save_dir_2', type = str, help = 'dir to save result ckpt files', default = 'checkpoints_2_resnet80_rsa/')
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 32#128
learning_rate = args.lr 

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
	alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
	beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
	for param_group in optimizer.param_groups:
		param_group['lr']=alpha_plan[epoch]
		param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
		
def accuracy(logit, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	output = F.log_softmax(logit, dim=1)
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title, fontsize=16)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")


	plt.ylabel('True label', fontsize=18)
	plt.xlabel('Predicted label', fontsize=18)
	plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Evaluate the Model
def evaluate(test_loader, model1):
	correct1 = 0
	total1 = 0
	a=np.zeros((3589))
	b=np.zeros((3589))
	i=0
	labels_sum=np.zeros(3589)
	for images, labels in test_loader:
		images = Variable(images).cuda()
		logits1 = model1(images)
		outputs1 = F.log_softmax(logits1, dim=1)
		_, pred1 = torch.max(outputs1.data, 1)
		a[i:i+len(labels)]=pred1.cpu()
		b[i:i+len(labels)]=labels
		total1 += labels.size(0)
		correct1 += (pred1.cpu() == labels).sum()
		labels_sum[i:i+len(labels)]=labels.numpy()
		i+=len(labels)
	
	matrix = confusion_matrix(b,a)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure(figsize=(10, 8))
	plot_confusion_matrix(matrix, classes=class_names, normalize=True,
						title= ' Confusion Matrix ' )
	plt.savefig('cm.png')
	plt.close()

	acc1 = 100*float(correct1)/float(total1)
	return acc1,a.T,labels_sum

def save_checkpoint(checkpoints_dir, model, optimizer, epoch):
	model_state_file = os.path.join(checkpoints_dir, 'model_state_{:02}.pytar'.format(epoch))
	optim_state_file = os.path.join(checkpoints_dir, 'optim_state_{:02}.pytar'.format(epoch))
	torch.save(model.state_dict(), model_state_file)
	torch.save(optimizer.state_dict(), optim_state_file)

def main():
	# Data Loader (Input Pipeline)
	print('loading dataset...')
	transform_train = transforms.Compose([
		transforms.Resize(100),
		transforms.RandomResizedCrop(90),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
		transforms.ToGray()
	])

	transform_test = transforms.Compose([
		transforms.Resize(100),
		transforms.CenterCrop(90),
		#transforms.Crop(x,y,90,90),
		transforms.ToTensor(),
		transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
		transforms.ToGray()
	])
	trainset = FER2013(split = 'Training', transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
	#PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
	#PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=37, shuffle=False, num_workers=1)
	PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
	test_loader = torch.utils.data.DataLoader(PrivateTestset, batch_size=37, shuffle=False, num_workers=1)
	# Define models
	print('building model...')
	cnn1 = ExpressionNet()
	cnn1.cuda()
	cnn1.eval()
	cnn1.load_state_dict(torch.load('/home/tiany/wsj/checkpoints_2/model_state_190.pytar'))
	test_acc1,a,labels=evaluate(test_loader, cnn1)
	print('Test Accuracy on the %s test images: Model1 %.4f %%' % (len(a), test_acc1))

if __name__=='__main__':
	main()
