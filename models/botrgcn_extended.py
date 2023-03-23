import torch
from torch import nn
from torch_geometric import nn as gnn

class BotRGCN(nn.Module):
    def __init__(self, desc_embedding_size=768, tweet_embedding_size=768, num_feature_size=5, 
                 cat_feature_size=3, embedding_dimension=128, num_relations=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # TODO: use torch_geometric.nn.Sequential instead?
        
        # user description layer
        self.desc_layer = nn.Sequential(
            nn.Linear(desc_embedding_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        # user tweet layer
        self.tweet_layer = nn.Sequential(
            nn.Linear(tweet_embedding_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        # numeric feature layer
        self.num_feature_layer = nn.Sequential(
            nn.Linear(num_feature_size, int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        # categorical feature layer
        self.cat_feature_layer = nn.Sequential(
            nn.Linear(cat_feature_size, int(embedding_dimension/4)),
            nn.LeakyReLU()  
        )
        
        self.inner = gnn.Sequential('x, edge_index, edge_type', [
            (nn.Linear(embedding_dimension, embedding_dimension), 'x -> x'),
            (nn.LeakyReLU(), 'x -> x'),
            (gnn.RGCNConv(embedding_dimension, embedding_dimension, num_relations=num_relations, aggr='max'), 'x, edge_index, edge_type -> x'),
            (gnn.LayerNorm(embedding_dimension), 'x -> x'),
            (nn.Dropout(self.dropout), 'x -> x'),
            (gnn.RGCNConv(embedding_dimension, embedding_dimension, num_relations=num_relations, aggr='max'), 'x, edge_index, edge_type -> x'),
            (gnn.LayerNorm(embedding_dimension), 'x -> x'),
            (nn.Linear(embedding_dimension, embedding_dimension), 'x -> x'),
            (nn.LeakyReLU(), 'x -> x'),
            (nn.Linear(embedding_dimension, 1), 'x -> x'),
            (nn.Sigmoid(), 'x -> x'),
            ])
        
        '''
        # embedding layer
        self.embedding_input_layer = nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()  
        )
        
        # RGCN layer
        # TODO: replace with FastRGCNConv?
        self.rgcn_layer = gnn.RGCNConv(embedding_dimension, embedding_dimension, num_relations=num_relations)
        
        # embedding output layer
        self.embedding_output_layer_1 = nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()  
        )
        
        # output layer
        self.output_layer = nn.Linear(embedding_dimension, 2)
        '''
        
    def forward(self, desc_embedding, tweet_embedding, num_feature, cat_feature, edge_index, edge_type):        
        desc = self.desc_layer(desc_embedding)
        tweets = self.tweet_layer(tweet_embedding)
        numeric = self.num_feature_layer(num_feature)
        cat = self.cat_feature_layer(cat_feature)
        x = torch.cat([desc, tweets, numeric, cat], dim=1)
        
        x = self.inner(x, edge_index, edge_type).flatten()
        return x
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)