import torch
from torch import nn

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm

from dataset import Twibot22
from models.botrgcn import BotRGCN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_size = 128
dropout = 0.3
lr = 1e-3
weight_decay = 5e-3

dataset = Twibot22(device=device)
model = BotRGCN(desc_embedding_size=768, tweet_embedding_size=768, num_feature_size=5, 
                 cat_feature_size=3, embedding_dimension=embedding_size, num_relations=2, dropout=dropout)
model.apply(model.init_weights)

model_metrics = MetricCollection([
    BinaryAccuracy(), 
    BinaryPrecision(), 
    BinaryRecall(), 
    BinaryF1Score()])

model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def main():
    losses = []
    for idx, e in tqdm(enumerate(range(0, 200))):
        loss = train()
        losses.append(loss.item())
    metrics = test()
    print(metrics)

    torch.cuda.empty_cache()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(
        dataset.description_embeddings,
        dataset.tweet_embeddings, 
        dataset.numerical_features,
        dataset.categorical_features.to(torch.float32),
        dataset.edge_index,
        dataset.edge_type)
    #print(out[dataset.train_mask][0:50])
    #print(dataset.labels[dataset.train_mask][0:50])
    loss = criterion(out[dataset.train_mask], dataset.labels[dataset.train_mask].to(torch.float32))
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    out = model(
        dataset.description_embeddings,
        dataset.tweet_embeddings, 
        dataset.numerical_features,
        dataset.categorical_features.to(torch.float32),
        dataset.edge_index,
        dataset.edge_type)
    #preds = torch.argmax(out[dataset.test_mask], dim=1).to('cpu').detach()
    preds = out[dataset.test_mask].to('cpu').detach()
    labels = dataset.labels[dataset.test_mask].to('cpu').detach()
    metrics = model_metrics(preds, labels)
    return metrics


if __name__ == "__main__":
    main()