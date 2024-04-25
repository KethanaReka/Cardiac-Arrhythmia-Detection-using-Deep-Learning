from __future__ import print_function
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

is_cuda = False
num_epochs = 20
batch_size = 10
torch.manual_seed(46)
log_interval = 10
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600
num_records = 48
num_classes = 16
allow_label_leakage = True

device = torch.device("cuda:2" if is_cuda else "cpu")


class CustomDatasetFromCSV(Dataset):
    def __init__(self, data_path, transforms_=None):
        self.df = pd.read_pickle(data_path)
        self.transforms = transforms_

    def __getitem__(self, index):
        row = self.df.iloc[index]
        signal = row['signal']
        target = row['target']
        if self.transforms is not None:
            signal = self.transforms(signal)
        signal = signal.reshape(1, signal.shape[0])
        return signal, target

    def __len__(self):
        return self.df.shape[0]


train_dataset = CustomDatasetFromCSV('./data/Arrhythmia_dataset.pkl')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def basic_layer(in_channels, out_channels, kernel_size, batch_norm=False, max_pool=True, conv_stride=1, padding=0
                , pool_stride=2, pool_size=2):
    layers = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=conv_stride,
                        padding=padding), nn.ReLU()]
    if batch_norm:
        layers.append(nn.BatchNorm1d(num_features=out_channels))
    if max_pool:
        layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))
    return nn.Sequential(*layers)


class arrhythmia_classifier(nn.Module):
    def __init__(self, in_channels=in_channels_):
        super(arrhythmia_classifier, self).__init__()
        self.cnn = nn.Sequential(
            basic_layer(in_channels=in_channels, out_channels=128, kernel_size=50, batch_norm=True, max_pool=True,
                        conv_stride=3, pool_stride=3),
            basic_layer(in_channels=128, out_channels=32, kernel_size=7, batch_norm=True, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=32, out_channels=32, kernel_size=10, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=32, out_channels=128, kernel_size=5, batch_norm=False, max_pool=True,
                        conv_stride=2, pool_stride=2),
            basic_layer(in_channels=128, out_channels=256, kernel_size=15, batch_norm=False, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=256, out_channels=512, kernel_size=5, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=512, out_channels=128, kernel_size=3, batch_norm=False, max_pool=False,
                        conv_stride=1),
            Flatten(),
            nn.Linear(in_features=1152, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, ex_features=None):
        return self.cnn(x)


model = arrhythmia_classifier().to(device).double()
lr = 0.0003

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.NLLLoss()


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


import matplotlib.pyplot as plt

def test():
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    sensitivity = recall_score(all_targets, all_preds, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')

    return accuracy  # Return accuracy for plotting

if __name__ == "__main__":
    train_losses = []
    test_accuracies = []  # List to store testing accuracies
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        accuracy = test()  # Call test() to get accuracy
        test_accuracies.append(accuracy)  # Append accuracy to list

    print("Evaluation on Test Set:")
    # Plotting the testing accuracies
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy Over Epochs')
    plt.legend()
    plt.show()
