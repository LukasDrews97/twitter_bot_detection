import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryMatthewsCorrCoef, BinaryConfusionMatrix, BinaryROC, BinaryAUROC
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from dataset import Twibot22
from models.botrgcn import BotRGCN
#from models.botrgcn_extended import BotRGCN

experiment_name = 'Experiment_2_5_all_relations'
preprocessed_data_folder = "./preprocessed_full/"
result_folder = 'results_full/'
edge_index_file = 'edge_index_full_95.pt'
edge_type_file = 'edge_type_full_95.pt'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Training parameters
number_of_runs = 3
random_seeds = [100, 200, 300, 400, 500]
embedding_size = 128 # Default: 128
dropout = 0.3
lr = 1e-5 # Default: 1e-5
weight_decay = 5e-3 # Default: 5e-3
batch_size=1024 # Default: 1024
training_epochs = 1000
num_neighbors = [256] * 4
num_relations = 5
roc_thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

dataset = Twibot22(root=preprocessed_data_folder, device='cpu', edge_index_file=edge_index_file, edge_type_file=edge_type_file)
# Model parameters
model_params = {'desc_embedding_size': 768, 'tweet_embedding_size': 768, 'num_feature_size': 5, 'cat_feature_size': 3, 'embedding_dimension': embedding_size, 'num_relations': num_relations, 'dropout': dropout}

# Define metrics
model_metrics = MetricCollection([
    BinaryAccuracy(),
    BinaryPrecision(),
    BinaryRecall(),
    BinaryF1Score(),
    BinaryMatthewsCorrCoef(),
    BinaryConfusionMatrix(),
    BinaryROC(thresholds=roc_thresholds),
    BinaryAUROC(thresholds=roc_thresholds)]
    )

data = dataset[0]
# Define neighborloaders for mini-batch sampling
train_loader = None
test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.test_mask)
validation_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.val_mask)

def main():
    time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    test_metrics = defaultdict(list)
    os.makedirs(result_folder, exist_ok=True)
    experiment_folder = f"{experiment_name}_{time}/"
    os.makedirs(result_folder + experiment_folder, exist_ok=True)

    # Run training for multiple runs 
    for run in range(1, number_of_runs+1):
        set_seed(random_seeds[run-1])
        global train_loader
        train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.train_mask, shuffle=True)
        model = init_model(model_params)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        losses = []
        validation_metrics = defaultdict(list)
        
        for e in range(training_epochs+1):
            # Calculate loss
            loss = train_epoch(model, optimizer, criterion)
            losses.append(loss)
        
            # calculate metrics on validation set every 10 epochs
            if e % 10 == 0:
                metrics = evaluate_on_validation_set(model)
                validation_metrics['epoch'].append(e)
                for k, v in metrics.items():
                    if k == 'BinaryConfusionMatrix':
                        validation_metrics['TrueNegatives'].append(v[0][0].item())
                        validation_metrics['FalseNegatives'].append(v[0][1].item())
                        validation_metrics['FalsePositives'].append(v[1][0].item())
                        validation_metrics['TruePositives'].append(v[1][1].item())
                    elif k == 'BinaryROC':
                        validation_metrics['roc_fpr'].append(";".join(str(x.item()) for x in v[0]))
                        validation_metrics['roc_tpr'].append(";".join(str(x.item()) for x in v[1]))
                        validation_metrics['roc_tresholds'].append(";".join(str(x.item()) for x in v[2]))
                    else:
                        validation_metrics[k].append(v.item())
            
                # calculate balanced accuracy
                tpr = validation_metrics['BinaryRecall'][-1]
                tnr = validation_metrics['TrueNegatives'][-1] / (validation_metrics['TrueNegatives'][-1] + validation_metrics['FalsePositives'][-1])
                balanced_accuracy = (tpr + tnr) / 2
                validation_metrics['BalancedBinaryAccuracy'].append(balanced_accuracy)

                print("===============================================")
                print(f"Run: {run}/{number_of_runs}, Epoch: {e}, Loss: {loss:.2f}")
                print(f"Accuracy: {metrics['BinaryAccuracy']:.2f}")
                print(f"BalancedAccuracy: {validation_metrics['BalancedBinaryAccuracy'][-1]:.2f}")
                print(f"Precision: {metrics['BinaryPrecision']:.2f}")
                print(f"Recall: {metrics['BinaryRecall']:.2f}")
                print(f"F1-Score: {metrics['BinaryF1Score']:.2f}")
                print(f"MCC: {metrics['BinaryMatthewsCorrCoef']:.2f}")
                print("===============================================")
    
        # calculate metrics on test set after training
        metrics = evaluate_on_test_set(model)
        test_metrics['run'].append(run)
        test_metrics['seed'].append(random_seeds[run-1])
        for k, v in metrics.items():
            if k == 'BinaryConfusionMatrix':
                test_metrics['TrueNegatives'].append(v[0][0].item())
                test_metrics['FalseNegatives'].append(v[0][1].item())
                test_metrics['FalsePositives'].append(v[1][0].item())
                test_metrics['TruePositives'].append(v[1][1].item())
            elif k == 'BinaryROC':
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

        print("===============================================")
        print("===============================================")
        print(f"Test Metrics, Run: {run}/{number_of_runs}, Seed: {random_seeds[run-1]}")
        print(f"Accuracy: {metrics['BinaryAccuracy']:.2f}")
        print(f"BalancedAccuracy: {test_metrics['BalancedBinaryAccuracy'][-1]:.2f}")
        print(f"Precision: {metrics['BinaryPrecision']:.2f}")
        print(f"Recall: {metrics['BinaryRecall']:.2f}")
        print(f"F1-Score: {metrics['BinaryF1Score']:.2f}")
        print(f"MCC: {metrics['BinaryMatthewsCorrCoef']:.2f}")
        print("===============================================")
        print("===============================================")

        # save loss, metrics and state dict
        validation_metrics_df = pd.DataFrame(validation_metrics)    
        losses_df = pd.DataFrame(losses, columns=['BCE loss'])
        validation_metrics_df.to_csv(f'{result_folder}{experiment_folder}{experiment_name}_run_{run}_validation_metrics_{time}.csv')
        losses_df.to_csv(f'{result_folder}{experiment_folder}{experiment_name}_run_{run}_losses_{time}.csv')
        torch.save(model.state_dict(), f'{result_folder}{experiment_folder}{experiment_name}_run_{run}_state_dict_{time}.pt')
    
    # save test metrics
    test_metrics_df = pd.DataFrame(test_metrics) 
    test_metrics_df.to_csv(f'{result_folder}{experiment_folder}{experiment_name}_test_metrics_{time}.csv')
    
    # calculate average and standard deviation
    print(test_metrics_df.describe())
    test_metrics_df.describe().to_csv(f'{result_folder}{experiment_folder}{experiment_name}_test_metrics_aggregated_{time}.csv')


def init_model(model_params):
    model = BotRGCN(**model_params)
    model.apply(model.init_weights)
    model.to(device)
    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_epoch(model, optimizer, loss_func):
    '''Train for one epoch.'''
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
        loss = loss_func(out[:batch_size], batch.y[:batch_size].to(torch.float32))
        average_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    
    average_loss /= len(train_loader) * batch_size
        
    return average_loss

def evaluate_on_test_set(model):
    '''Evaluate on test set.'''
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

def evaluate_on_validation_set(model):
    '''Evaluate on validation set.'''
    model.eval()
    
    labels = []
    predictions = []
    
    with torch.no_grad():
        for batch in validation_loader:
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