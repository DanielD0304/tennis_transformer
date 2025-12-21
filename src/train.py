import torch
from torch.utils.data import DataLoader
from .dataset import DataSet
from .model import Model

def train():
    num_epochs = 10
    batch_size = 32
    d_model = 64
    num_heads = 4
    num_layers = 2
    learning_rate = 0.001


if __name__ == "__main__":
    train()