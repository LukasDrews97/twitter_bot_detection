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
        self.labels = torch.load(path("labels.pt"), map_location=self.device)
        
        # load node features
        self.numerical_features = torch.load(path("num_properties_tensor.pt"), map_location=self.device)
        self.categorical_features = torch.load(path("categorical_properties_tensor.pt"), map_location=self.device)
        self.description_embeddings = torch.load(path("user_description_embedding_tensor.pt"), map_location=self.device)
        self.tweet_embeddings = torch.load(path("user_tweets_tensor.pt"), map_location=self.device)
        self.merged_features = torch.cat([self.numerical_features, self.categorical_features, self.description_embeddings, self.tweet_embeddings], dim=1)
        
        # load edge index and types
        self.edge_index = torch.load(path("edge_index.pt"), map_location=self.device)
        self.edge_type = torch.load(path("edge_type.pt"), map_location=self.device)
        
        # load dataset masks
        self.train_mask = torch.load(path("train_mask.pt"), map_location=self.device)
        self.test_mask = torch.load(path("test_mask.pt"), map_location=self.device)
        self.val_mask = torch.load(path("validation_mask.pt"), map_location=self.device)
        
        # create data object
        self.data = Data(x=self.merged_features, edge_index=self.edge_index, edge_attr=self.edge_type, y=self.labels).to(self.device)
        self.data.train_mask = self.train_mask
        self.data.test_mask = self.test_mask
        self.data.val_mask = self.val_mask
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
