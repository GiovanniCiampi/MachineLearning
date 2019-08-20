import numpy as np
import pandas as pd
from .Node import Node

def get_gini_impurity(data, label_column):
    
    labels = set(data[label_column])
    probabilities = [sum(data[label_column] == label)/float(len(data)) for label in labels]
        
    impurity = 1
    for probability in probabilities:
        impurity -= probability**2
        
    return impurity


def get_information_gain(data_split_L, data_split_R, label_column, current_impurity):
    
    dataset_length = len(data_split_L)+ len(data_split_R)
    
    gini_L = get_gini_impurity(data_split_L, label_column)
    frac_L = len(data_split_L)/dataset_length

    gini_R = get_gini_impurity(data_split_R, label_column)
    frac_R = len(data_split_R)/dataset_length
    
    return current_impurity - (gini_L * frac_L + gini_R * frac_R)


def is_non_separable(data, label_column):
    
    attributes = list(data.columns)
    attributes.remove(label_column)
    
    for attribute in attributes:
        if len(set(data[attribute].values)) > 1:
            return False
    return True


def get_next_split(data, label_column):
    current_impurity = get_gini_impurity(data, label_column)
    
    if current_impurity <= 0 or is_non_separable(data, label_column):
        return [current_impurity]
    
    best_information_gain = -1
    best_split_attribute = None
    best_split_value = None
    
    attributes = list(data.columns)
    attributes.remove(label_column)

    for attribute in attributes:
        values = set(data[attribute].values)
        
        for value in values:
            split_L = data[data[attribute] == value]
            split_R = data[data[attribute] != value]
            
            ig = get_information_gain(split_L, split_R, label_column, current_impurity)
            if ig > best_information_gain:
                best_information_gain = ig
                best_split_attribute = attribute
                best_split_value = value
            
    return current_impurity, best_split_attribute, best_split_value

            
def build_tree(data, label_column):
    split = get_next_split(data, label_column)
    impurity = split[0]
        
    if len(split) < 2:
        decision = data[label_column].mode()[0]
        return Node(is_leaf=True, impurity=impurity, decision=decision)
    
    
    split_attribute = split[1]
    split_value = split[2]
    
    split_L = data[data[split_attribute] == split_value]
    split_R = data[data[split_attribute] != split_value]
    
    return Node(impurity=impurity, left_child=build_tree(split_L, label_column),
                right_child=build_tree(split_R, label_column),
                split_attribute=split_attribute, split_value=split_value)