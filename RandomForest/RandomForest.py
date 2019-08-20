import DecisionTree.Node

import pandas as pd
import numpy as np

import random

from scipy.stats import mode
from DecisionTree.DecisionTree import build_tree

class RandomForest:
    def __init__(self, trees):
        self.trees=trees
        self.n_trees = len(trees)
    
    def predict(self, data):
        raw_predictions=[]
        final_predictions=[]
        
        for tree in self.trees:
            raw_predictions.append(np.asarray(tree.predict(data))[:, 0])
        raw_predictions = np.asarray(raw_predictions).transpose()
        
        for idx, row in enumerate(raw_predictions):
            pred = mode(row)
            final_predictions.append([pred[0][0], pred[1][0]/float(self.n_trees)])
        return final_predictions
    
def build_forest(attributes_sampling_rate, data_sampling_rate, n_trees, data, label_column):
    
    attributes = data.columns.tolist()
    attributes.remove(label_column)
    
    attributes_sampling_rate = int(attributes_sampling_rate*len(attributes))
    data_sampling_rate = int(data_sampling_rate*len(data))
    
    trees = []
    
    for i in range(n_trees):
        current_attributes = random.sample(attributes, attributes_sampling_rate)
        current_data = data.sample(frac=1).reset_index(drop=True)[:data_sampling_rate]
        
        current_data = current_data[current_attributes+[label_column]]
        
        trees.append(build_tree(current_data, label_column))
    return RandomForest(trees)