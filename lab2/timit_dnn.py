# Train a torch DNN for Kaldi DNN-HMM model

import math
import sys

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dnn.torch_dataset import TorchSpeechDataset
from dnn.torch_dnn import TorchDNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 2
HIDDEN_DIM = 256
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 50
PATIENCE = 3

if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = sys.argv[1]


# FIXME: You may need to change these paths
TRAIN_ALIGNMENT_DIR = "exp_tri1_ali_train"
DEV_ALIGNMENT_DIR = "exp_tri1_ali_dev"
TEST_ALIGNMENT_DIR = "exp_tri1_ali_test"


def train(model, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    """Train model using Early Stopping and save the checkpoint for
    the best validation loss
    """
    # TODO: IMPLEMENT THIS FUNCTION
    raise NotImplementedError


trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')

scaler = StandardScaler()
scaler.fit(trainset.feats)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1))


dnn = TorchDNN(
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P
)
dnn.to(DEVICE)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
dev_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

optimizer = ...
criterion = ...

train(model, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
