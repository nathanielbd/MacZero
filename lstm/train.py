import pickle
import numpy as np
from sklearn.decomposition import PCA

with open('../Data/traindata.pkl', 'rb') as f:
    train_inputs, train_outputs, test_inputs, test_outputs, dev_inputs, dev_outputs = pickle.load(f)

train_inputs, test_inputs, dev_inputs = train_inputs/255, test_inputs/255, dev_inputs/255

idx = np.random.choice(train_inputs.shape[0], 1000, replace=False)
pca = PCA(n_components=64)
pca.fit(train_inputs[idx])
train_inputs = pca.transform(train_inputs)
test_inputs = pca.transform(test_inputs)
dev_inputs = pca.transform(dev_inputs)

from joblib import dump, load
dump(pca, 'pca.joblib')
pca_fromload = load('pca.joblib')

import torch
import torch.nn as nn
import torch.optim as optim

INPUT_SIZE = 64
OUTPUT_SIZE = 3
LAYERS = 2
model = nn.LSTM(INPUT_SIZE, OUTPUT_SIZE, LAYERS).float()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs = 10

class PO_Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, seq_len):
        self.seq_len = seq_len
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs)-self.seq_len

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.inputs[index:index+self.seq_len]),
            torch.from_numpy(self.outputs[index:index+self.seq_len])
        )


# train_dataset = list(zip(torch.from_numpy(train_inputs), torch.from_numpy(train_outputs)))
SEQ_LEN = 40
train_dataset = PO_Dataset(train_inputs, train_outputs, SEQ_LEN)
BATCH_SIZE = 200
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

losses = []
for epoch in range(epochs):
    loss = 0
    for data in train_loader:
        inputs, moves = data
        # inputs = torch.reshape(inputs, (SEQ_LEN, BATCH_SIZE//SEQ_LEN, INPUT_SIZE))
        # moves = torch.reshape(moves, (SEQ_LEN, BATCH_SIZE//SEQ_LEN, OUTPUT_SIZE))
        inputs, moves = inputs.float(), moves.float()
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        train_loss = criterion(outputs.transpose(1,2), moves.max(dim=2)[1])
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    losses.append(loss)
    # if epoch%100==0:
    print(f"epoch: {epoch}/{epochs}, loss: {loss}")

torch.save(model.state_dict(), 'lstm.pt')

import matplotlib.pyplot as plt
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()