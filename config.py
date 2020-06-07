import numpy as np
import argparse


def get_args():
	parser = argparse.ArgumentParser('Auto-Augment')
	# cnn model
	parser.add_argument('--layers', type=int, default=40)
	parser.add_argument('--widening_factor', type=int, default=2)
	parser.add_argument('--dropout', type=float, default=0.3)
	parser.add_argument('--cnn_train_epochs', type=int, default=120)    # 120
	parser.add_argument('--cnn_lr', type=float, default=0.025)
	parser.add_argument('--cnn_weight_decay', type=float, default=3e-4)
	
	# dataset
	parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar')
	parser.add_argument('--batch_size', type=int, default=128)
	
	# search space
	parser.add_argument('--augment_types', type=list, default=['shearX', 'shearY', 'translateX', 'translateY',
	                                        'rotate', 'color', 'posterize', 'solarize', 'contrast', 'sharpness',
	                                        'brightness', 'autocontrast', 'equalize', 'invert'
											], help='all searched policies')
	parser.add_argument('--magnitude_types', type=list, default=range(10))
	parser.add_argument('--prob_types', type=list, default=range(11))
	parser.add_argument('--op_num_pre_subpolicy', type=int, default=2)
	parser.add_argument('--subpolicy_num', type=int, default=5)
	
	# controller
	parser.add_argument('--controller_hid_size', type=int, default=100)
	parser.add_argument('--controller_lr', type=float, default=3.5e-4)
	parser.add_argument('--softmax_temperature', type=float, default=5.)
	parser.add_argument('--tanh_c', type=float, default=2.5)
	parser.add_argument('--entropy_coeff', type=float, default=1e-5)
	parser.add_argument('--baseline_decay', type=float, default=0.95)
	parser.add_argument('--controller_grad_clip', type=float, default=0.)
	
	# training
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--search_epochs', type=int, default=1500)  # 1500
	
	arguments = parser.parse_args()
	print(arguments)
	return arguments
	
















