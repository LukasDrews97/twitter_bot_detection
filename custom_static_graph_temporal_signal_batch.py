import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Batch

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Batch_Size = int
Num_Nodes = int
Additional_Features = List[np.ndarray]


class CustomStaticGraphTemporalSignalBatch(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Batch object. Between two
    temporal snapshots the feature matrix, target matrices and optionally passed
    attributes might change. However, the underlying graph is the same.

    Args:
        edge_index (Torch array): Index tensor of edges.
        edge_weight (Torch array): Edge weight tensor.
        features (List of Torch arrays): List of node feature tensors.
        targets (List of Torch arrays): List of node label (target) tensors.
        batch_size (int): Batch size.
        **kwargs (optional List of Torch arrays): List of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        features: Node_Features,
        targets: Targets,
        batch_size: Batch_Size,
        num_nodes: Num_Nodes,
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self):
        return self.edge_index
    
    def _get_edge_weight(self):
        return self.edge_weight

    def _get_feature(self, time_index: int):
        return self.features[time_index]

    def _get_target(self, time_index: int):
        return self.targets[time_index]

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        return feature

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features
    
    def _get_batches(self):
        batches = []
        for i in range(self.batch_size):
            batches.append(np.array([i for _ in range(self.num_nodes)]))
        batches = np.concatenate(batches)
        return batches

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = CustomStaticGraphTemporalSignalBatch(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                self.batch_size,
                self.num_nodes,
                **{key: getattr(self, key) for key in self.additional_feature_keys}
            )
        else:
            x = self._get_feature(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)

            snapshot = Batch(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, batch=self._get_batches(), **{key: getattr(self, key) for key in self.additional_feature_keys})
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self