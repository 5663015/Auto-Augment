import argparse
from utils import *
import torch.nn as nn
from thop import profile
from model import WideResNet
from datetime import datetime
import torch.nn.functional as F
from torchsummary import summary
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import time


parser = argparse.ArgumentParser('Train signal model')
parser.add_argument('--exp_name', type=str, help='search model name')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--seed', type=int, default=2020, help='seed')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--random_id', type=int, default=0, help='random_id')
# ******************************* dataset *******************************#
parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, imagenet]')
parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar', help='dataset dir')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

args = parser.parse_args()
print(args)


def train(args, epoch, train_data, device, model, criterion, optimizer):
	model.train()
	train_loss = 0.0
	top1 = AvgrageMeter()
	top5 = AvgrageMeter()

	for step, (inputs, targets) in enumerate(train_data):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		
		outputs= model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		
		nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
		n = inputs.size(0)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)
		
		optimizer.step()
		train_loss += loss.item()
		print('\rEpoch: {}, step: {}/{}, train loss: {:.6}, top1: {:.6}, top5: {:.6}'.format(
		        epoch, step, len(train_data), loss, top1.avg, top5.avg), end='')

	return train_loss/(step+1), top1.avg, top5.avg


def validate(val_data, device, model):
	model.eval()
	val_loss = 0.0
	val_top1 = AvgrageMeter()
	val_top5 = AvgrageMeter()
	criterion = nn.CrossEntropyLoss().to(device)

	with torch.no_grad():
		for step, (inputs, targets) in enumerate(val_data):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			val_loss += loss.item()
			prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
			n = inputs.size(0)
			val_top1.update(prec1.item(), n)
			val_top5.update(prec5.item(), n)

	return val_top1.avg, val_top5.avg, val_loss / (step + 1)


def main():
	if not torch.cuda.is_available():
		device = torch.device('cpu')
	else:
		torch.cuda.set_device(args.gpu)
		cudnn.benchmark = True
		cudnn.enabled = True
		device = torch.device("cuda")
	
	criterion = nn.CrossEntropyLoss().to(device)
	
	model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3)
	model = model.to(device)
	summary(model, (3, 32, 32))
	
	optimizer = torch.optim.SGD(
	    model.parameters(),
	    args.learning_rate,
	    momentum=args.momentum,
	    weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
	    optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1)
	
	train_transform, valid_transform = data_transforms_cifar(args)
	trainset = dset.CIFAR10(root=args.data_dir, train=True, download=False, transform=train_transform)
	train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
	                                          shuffle=True, pin_memory=True, num_workers=8)
	valset = dset.CIFAR10(root=args.data_dir, train=False, download=False, transform=valid_transform)
	valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
	                                          shuffle=False, pin_memory=True, num_workers=8)

	best_acc = 0.0
	for epoch in range(args.epochs):
		t1 = time.time()
		
		# train
		train(args, epoch, train_queue, device, model, criterion=criterion, optimizer=optimizer)
		lr = scheduler.get_lr()[0]
		scheduler.step()
		
		# validate
		val_top1, val_top5, val_obj = validate(val_data=valid_queue, device=device, model=model)
		if val_top1 > best_acc:
			best_acc = val_top1
		t2 = time.time()
		
		print('\nval: loss={:.6}, top1={:.6}, top5={:.6}, lr: {:.8}, time: {:.4}'
		      .format(val_obj, val_top1, val_top5, lr, t2-t1))
		print('Best Top1 Acc: {:.6}'.format(best_acc))


if __name__ == '__main__':
	main()

