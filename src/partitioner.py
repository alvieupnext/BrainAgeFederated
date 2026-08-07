from typing import Dict, List
import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner

from distributions import dataframes_from_distribution

class BrainAgePartitioner(Partitioner):
    """
    Custom Flower Datasets Partitioner that wraps the highly specific
    Gaussian/Transition/Original distribution logic for the Brain Age project.
    """
    def __init__(self, distribution_type: str, num_partitions: int):
        super().__init__()
        self._num_partitions = num_partitions
        self.distribution_type = distribution_type
        self._indices_per_partition: Dict[int, List[int]] = {}

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def load_partition(self, partition_id: int) -> Dataset:
        # 1. Ensure the dataset has been assigned (via partitioner.dataset = dataset)
        if not self.is_dataset_assigned():
            raise ValueError("Dataset is not assigned to the partitioner.")
            
        # 2. On the first call, compute the complex distribution logic
        if not self._indices_per_partition:
            # Convert to Pandas for compatibility with distributions.py
            df = self.dataset.to_pandas()
            
            # Inject a column to track original row indices since the
            # internal logic uses ignore_index=True when concatenating
            df['_row_idx'] = np.arange(len(df))
            
            # Call the custom logic
            results = dataframes_from_distribution(df, self.distribution_type, self._num_partitions)
            
            # dataframes_from_distribution returns a dict: { 'node_name': dataframe }
            # We map these to partition_ids 0 -> num_partitions-1
            for i, (name, part_df) in enumerate(results.items()):
                self._indices_per_partition[i] = part_df['_row_idx'].astype(int).tolist()

        # 3. Use the computed row indices to select a subset from the HuggingFace Dataset
        partition_indices = self._indices_per_partition[partition_id]
        return self.dataset.select(partition_indices)
