import os
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryMatthewsCorrCoef, BinaryConfusionMatrix, BinaryROC
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from dataset import Twibot22
#from models.botrgcn import BotRGCN
from models.botrgcn_extended import BotRGCN

experiment_name = 'roc_test'
preprocessed_data_folder = "./preprocessed_full_sebastian/"
result_folder = 'results_full/'
edge_index_file = 'edge_index_retweeted.pt'
edge_type_file = 'edge_type_retweeted.pt'

#preprocessed_data_folder = "./preprocessed_subgraph_20230220/"
#result_folder = "./results_subgraph/"
#edge_index_file = 'edge_index_3rel.pt'
#edge_type_file = 'edge_type_3rel.pt'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_size = 128 # Default: 128
dropout = 0.3
lr = 1e-5 # Default: 1e-4
weight_decay = 5e-3
batch_size=1024 # Default: 1024
training_epochs = 2
num_neighbors = [256] * 4
num_relations = 2
roc_thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

dataset = Twibot22(root=preprocessed_data_folder, device='cpu', edge_index_file=edge_index_file, edge_type_file=edge_type_file)
model = BotRGCN(desc_embedding_size=768, tweet_embedding_size=768, num_feature_size=5, 
                 cat_feature_size=3, embedding_dimension=embedding_size, num_relations=num_relations, dropout=dropout)
model.apply(model.init_weights)
model.to(device)

model_metrics = MetricCollection([
    BinaryAccuracy(),
    BinaryPrecision(),
    BinaryRecall(),
    BinaryF1Score(),
    BinaryMatthewsCorrCoef(),
    BinaryConfusionMatrix(),
    BinaryROC(thresholds=roc_thresholds)])

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
                if k == 'BinaryConfusionMatrix':
                    test_metrics['TrueNegatives'].append(v[0][0].item())
                    test_metrics['FalseNegatives'].append(v[0][1].item())
                    test_metrics['FalsePositives'].append(v[1][0].item())
                    test_metrics['TruePositives'].append(v[1][1].item())
                elif k == 'BinaryROC':
                    fpr = v[0]
                    tpr = v[1]
                    th = v[2]
                    test_metrics['roc_fpr'].append(";".join(str(x.item()) for x in v[0]))
                    test_metrics['roc_tpr'].append(";".join(str(x.item()) for x in v[1]))
                    test_metrics['roc_tresholds'].append(";".join(str(x.item()) for x in v[2]))
                else:
                    test_metrics[k].append(v.item())
            
            # calculate balanced accuracy
            tpr = test_metrics['BinaryRecall'][-1]
            tnr = test_metrics['TrueNegatives'][-1] / (test_metrics['TrueNegatives'][-1] + test_metrics['FalsePositives'][-1])
            balanced_accuracy = (tpr + tnr) / 2
            test_metrics['BalancedBinaryAccuracy'].append(balanced_accuracy)

            print(f"Epoch: {e}, Loss: {loss:.2f}")
            print(f"Accuracy: {metrics['BinaryAccuracy']:.2f}")
            print(f"BalancedAccuracy: {test_metrics['BalancedBinaryAccuracy'][-1]:.2f}")
            print(f"Precision: {metrics['BinaryPrecision']:.2f}")
            print(f"Recall: {metrics['BinaryRecall']:.2f}")
            print(f"F1-Score: {metrics['BinaryF1Score']:.2f}")
            print(f"MCC: {metrics['BinaryMatthewsCorrCoef']:.2f}")
            print("===============================================")


    # save loss, metrics and state dict
    test_metrics_df = pd.DataFrame(test_metrics)     
    losses_df = pd.DataFrame(losses, columns=['BCE loss'])

    time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    os.makedirs(result_folder, exist_ok=True)

    test_metrics_df.to_csv(f'{result_folder}{experiment_name}_test_metrics_{time}.csv')
    losses_df.to_csv(f'{result_folder}{experiment_name}_losses_{time}.csv')
    torch.save(model.state_dict(), f'{result_folder}{experiment_name}_state_dict_{time}.pt')


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
    #true_positives = 0
    #true_negatives = 0
    #false_positives = 0
    #false_negatives = 0
    
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
            #out_2 = torch.greater(out, 0.5)
            #confusion_vector = batch.y[:batch_size].to('cpu') / out_2[:batch_size].to('cpu')
            #true_positives += torch.sum(confusion_vector == 1).item()
            #false_positives += torch.sum(confusion_vector == float('inf')).item()
            #true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
            #false_negatives += torch.sum(confusion_vector == 0).item()

            labels.extend(list(batch.y[:batch_size].to('cpu')))
            predictions.extend(list(out[:batch_size].to('cpu')))
        #print("tp:" + str(true_positives))
        #print("tn:" + str(true_negatives))
        #print("fp: " + str(false_positives))
        #print("fn:" + str(false_negatives))
        metrics = model_metrics(torch.tensor(predictions), torch.tensor(labels))
    return metrics


if __name__ == "__main__":
    main()