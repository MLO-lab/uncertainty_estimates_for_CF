import os, pickle
import torch
import numpy as np

from collections import defaultdict
from models.resnet import SlimResNet18
from models.mlp import MLP
from sklearn.metrics import balanced_accuracy_score
from torchvision import transforms
from torchvision.models import resnet18, resnet50 # the self-implemented resnet is too slow
from utils.data_loader import get_statistics
from utils.cl_utils import Client
from utils.utils_memory import Memory

def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used")
    memory_available = [int(x.split()[2]) for x in open("./utils/tmp", "r").readlines()]
    return np.argmin(memory_available)


def get_logger(args):
    _, _, _, _, _ = get_statistics(args)

    log = {}
    log['train'] = defaultdict(dict)
    for client_id in range(args.n_clients):
        log['train']['loss'][client_id] = np.zeros([args.n_tasks, args.n_runs])

    for mode in ['test', 'val']:
        log[mode] = defaultdict(dict)
        for client_id in range(args.n_clients):
            log[mode]['acc'][client_id] = np.zeros([args.n_runs, args.n_tasks, args.n_tasks])
            log[mode]['forget'][client_id] = np.zeros([args.n_runs])
            log[mode]['bal_acc'][client_id] = np.zeros([args.n_runs])

    log['global_test'] = defaultdict(dict)
    log['global_test']['bal_acc'] = np.zeros([args.n_runs])
    for task_id in range(args.n_tasks):
        log['global_test']['acc'] = np.zeros([args.n_runs, args.n_tasks, args.n_tasks])

    return log


def custom_resnet(args, model):
    model.conv1 = torch.nn.Conv2d(args.input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, args.n_classes)
    return model.to(args.device)


def initialize_model(args):
    if args.model_name == 'resnet18':
        model = resnet18().to(args.device)
    if args.model_name == 'resnet18_pre':
        resnet = resnet18(weights='DEFAULT')
        model = custom_resnet(args, resnet)
    if args.model_name == 'resnet50':
        model = resnet50().to(args.device)
    if args.model_name == 'resnet50_pre':
        resnet = resnet50(weights='DEFAULT')
        model = custom_resnet(args, resnet)
    if args.model_name == 'resnet':
        model = SlimResNet18(nclasses=args.n_classes, input_size=args.input_size).to(args.device)
    if args.model_name == 'mlp':
        model = MLP(args).to(args.device)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion


def save_results(args, logger):
    logger_fn = f'{args.dir_results}/logger.pkl'
    with open(logger_fn, 'wb') as outfile:
        pickle.dump(logger, outfile)
        outfile.close()