import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import WideResNet
from controller import Controller
from augement_policy import Policy
from config import *
from utils import *
import time


def get_data_loader(args, policy_provider):
	MEAN = [0.49139968, 0.48215827, 0.44653124]
	STD = [0.24703233, 0.24348505, 0.26158768]
	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD)
	])
	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD)
	])
	train_transform.transforms.insert(0, policy_provider)
	
	trainset = dset.CIFAR10(root=args.data_dir, train=True, download=False, transform=train_transform)
	train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
	                                          shuffle=True, pin_memory=True, num_workers=8)
	valset = dset.CIFAR10(root=args.data_dir, train=False, download=False, transform=valid_transform)
	valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
	                                          shuffle=False, pin_memory=True, num_workers=8)
	return train_queue, valid_queue, train_transform


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


def train_cnn(args, model, device, train_quene, val_quene):
	optimizer = torch.optim.SGD(model.parameters(), lr=args.cnn_lr, momentum=0.9, weight_decay=args.cnn_weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cnn_train_epochs, eta_min=1e-8)
	best_val_acc = 0.
	
	for e in range(args.cnn_train_epochs):
		model.train()
		scheduler.step()
		
		t1 = time.time()
		train_loss, top1, top5 = 0.0, AvgrageMeter(), AvgrageMeter()
		criterion = nn.CrossEntropyLoss().to(device)
		for step, (inputs, targets) in enumerate(train_quene):
			inputs, targets = inputs.to(device), targets.to(device)
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			
			# nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
			n = inputs.size(0)
			top1.update(prec1.item(), n)
			top5.update(prec5.item(), n)
			
			optimizer.step()
			train_loss += loss.item()
			
			print('\rEpoch: {}, step: {}/{}, train loss: {:.6}, top1: {:.4}, top5: {:.4}'.format(
					e+1, step, len(train_quene), loss, top1.avg, top5.avg), end='')
		val_acc, _, _ = validate(val_quene, device, model)
		if val_acc > best_val_acc:
			best_val_acc= val_acc
		t2 = time.time()
		print('\nval acc of this epoch: {:.4}, best val acc: {:.4}, time: {:.4}/s'.format(val_acc, best_val_acc, t2-t1))
	return best_val_acc


def train_controller(args, controller, optimizer, val_acc, baseline):
	controller.train()
	
	entropies, log_prob = controller.entropies, controller.log_probs
	# entropies, log_prob = torch.Tensor(np.array(entropies)).cuda(), torch.Tensor(np.array(log_prob)).cuda()
	# np_entropies = entropies.data.cpu().numpy()
	reward = val_acc + args.entropy_coeff * entropies
	
	if baseline is None:
		baseline = reward
	else:
		decay = args.baseline_decay
		baseline = decay * baseline + (1 - decay) * reward
		baseline = baseline.clone().detach()
	
	adv = reward - baseline
	# loss = -log_prob * get_variable(adv, args.cuda, requires_grad=False)
	loss = -log_prob * adv
	loss -= args.entropy_coeff * entropies
	loss = loss.sum()
	
	optimizer.zero_grad()
	loss.backward()
	
	if args.controller_grad_clip > 0:
		torch.nn.utils.clip_grad_norm(controller.parameters(), args.controller_grad_clip)
	optimizer.step()
	
	print('entropies: {}, log_prob: {}, reward: {}, loss: {}'.format(entropies.item(), log_prob.item(), reward, loss))
	

def main(args):
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	
	# controller
	controller = Controller(args).to(device)
	controller_optimizer = torch.optim.SGD(controller.parameters(), args.controller_lr, momentum=0.9)
	baseline = None
	
	# search
	for epoch in range(args.search_epochs):
		print('-'*50)
		print('{} th search'.format(epoch + 1))
		print('-'*50)
		
		# sample subpolicy
		print('*'*30)
		print('sample subpolicy')
		print('*'*30)
		controller.eval()
		policy_dict = controller.sample()
		policy_provider = Policy(args, policy_dict)
		for p in policy_dict:
			print(p)
		
		# get dataset
		train_queue, valid_queue, train_transform = get_data_loader(args, policy_provider)
		
		# train cnn
		print('*' * 30)
		print('train cnn')
		print('*' * 30)
		model = WideResNet(depth=args.layers, num_classes=10, widen_factor=args.widening_factor,
		                   dropRate=args.dropout).to(device)
		val_acc = train_cnn(args, model, device, train_queue, valid_queue)
		
		# train controller
		print('*' * 30)
		print('train controller')
		print('*' * 30)
		train_controller(args, controller, controller_optimizer, val_acc, baseline)
		
		# save
		state = {
			'args': args,
			'best_acc': val_acc,
			'controller_state': controller.state_dict(),
			'policy_dict': policy_dict
		}
		torch.save(state, './models/{}.pt.tar'.format(epoch))
		

if __name__ == '__main__':
	args = get_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	
	main(args)








