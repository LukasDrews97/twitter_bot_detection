import os
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryMatthewsCorrCoef
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from dataset import Twibot22
from models.botrgcn import BotRGCN

preprocessed_data_folder = './preprocessed_vm'
result_folder = 'results/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_size = 128
dropout = 0.1
lr = 1e-3
weight_decay = 5e-2
batch_size = 1024
training_epochs = 10
num_neighbors = [256] * 4

dataset = Twibot22(root=preprocessed_data_folder, device='cpu')
model = BotRGCN(desc_embedding_size=768, tweet_embedding_size=768, num_feature_size=5,
                 cat_feature_size=3, embedding_dimension=128, num_relations=2, dropout=dropout)
model.apply(model.init_weights)
model.to(device)

model_metrics = MetricCollection([
    BinaryAccuracy(),
    BinaryPrecision(),
    BinaryRecall(),
    BinaryF1Score(),
    BinaryMatthewsCorrCoef()])

model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

data = dataset[0]
train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.train_mask, shuffle=True)
test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.test_mask)
validation_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.val_mask)

def main():
    losses = []
    test_metrics = defaultdict(list)

    for e in range(training_epochs+1):
        loss = train_epoch()
        losses.append(loss)
        
        if e % 10 == 0:
            metrics = evaluate()
            test_metrics['epoch'].append(e)
            for k, v in metrics.items():
                test_metrics[k].append(v.item())
            
            print(f"Epoch: {e}, Loss: {loss:.2f}")
            print(f"Accuracy: {metrics['BinaryAccuracy']:.2f}")
            print(f"Precision: {metrics['BinaryPrecision']:.2f}")
            print(f"Recall: {metrics['BinaryRecall']:.2f}")
            print(f"F1-Score: {metrics['BinaryF1Score']:.2f}")
            print(f"MCC: {metrics['BinaryMatthewsCorrCoef']:.2f}")
            print()

        
    # save loss, metrics and state dict
    test_metrics_df = pd.DataFrame(test_metrics)     
    losses_df = pd.DataFrame(losses, columns=['BCE loss'])

    time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    os.makedirs(result_folder, exist_ok=True)

    test_metrics_df.to_csv(f'{result_folder}test_metrics_{time}.csv')
    losses_df.to_csv(f'{result_folder}losses_{time}.csv')
    torch.save(model.state_dict(), f'{result_folder}state_dict_{time}.pt')


def train_epoch():
    model.train()
    average_loss = 0.0
    
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(
            batch.description_embeddings,
            batch.tweet_embeddings, 
            batch.numerical_features,
            batch.categorical_features.to(torch.float32),
            batch.edge_index,
            batch.edge_attr)
        loss = criterion(out[:batch_size], batch.y[:batch_size].to(torch.float32))
        average_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    
    average_loss /= len(train_loader) * batch_size
        
    return average_loss

def evaluate():
    model.eval()
    
    labels = []
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            out = model(
                batch.description_embeddings,
                batch.tweet_embeddings, 
                batch.numerical_features,
                batch.categorical_features.to(torch.float32),
                batch.edge_index,
                batch.edge_attr)
            
            labels.extend(list(batch.y[:batch_size].to('cpu')))
            predictions.extend(list(out[:batch_size].to('cpu')))
        metrics = model_metrics(torch.tensor(predictions), torch.tensor(labels))
    return metrics


if __name__ == "__main__":
    main()