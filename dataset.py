import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

class Twibot22(Dataset):
    def __init__(self, root=r'src/Data_test/preprocessed', device='cpu'):
        self.root = root
        super().__init__(self.root, None, None, None)
        self.device = device
        path = lambda name: f"{self.root}/{name}"
        
        # load labels
        labels = torch.load(path("labels.pt"), map_location=self.device)
        
        # load node features
        numerical_features = torch.load(path("num_properties_tensor.pt"), map_location=self.device)
        categorical_features = torch.load(path("categorical_properties_tensor.pt"), map_location=self.device)
        description_embeddings = torch.load(path("user_description_embedding_tensor.pt"), map_location=self.device)
        tweet_embeddings = torch.load(path("user_tweets_tensor.pt"), map_location=self.device)
        #merged_features = torch.cat([numerical_features, categorical_features, description_embeddings, tweet_embeddings], dim=1)
        
        # load edge index and types
        edge_index = torch.load(path("edge_index.pt"), map_location=self.device)
        edge_type = torch.load(path("edge_type.pt"), map_location=self.device)
        
        # load dataset masks
        train_mask = torch.load(path("train_mask.pt"), map_location=self.device)
        test_mask = torch.load(path("test_mask.pt"), map_location=self.device)
        val_mask = torch.load(path("validation_mask.pt"), map_location=self.device)
        
        self.data = Data(
            edge_index=edge_index,
            edge_attr=edge_type,
            y=labels,
            description_embeddings = description_embeddings,
            tweet_embeddings = tweet_embeddings,
            numerical_features = numerical_features,
            categorical_features = categorical_features,
            train_mask = train_mask,
            test_mask = test_mask,
            val_mask = val_mask,
            num_nodes = labels.shape[0]
        )
        
        assert self.data.validate()
        
    def len(self):
        return 1
    
    def get(self, idx):
        if idx == 0: return self.data
    
    @property
    def num_node_features(self):
        return self.data.num_node_features
    
    @property
    def num_edge_features(self):
        return self.data.num_edge_features
    
    @property
    def num_nodes(self):
        return self.data.num_nodes
    
    @property
    def num_edges(self):
        return self.data.num_edges  
