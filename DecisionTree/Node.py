class Node:
    def __init__(self, impurity, is_leaf=False, left_child=None, right_child=None, split_attribute=None, split_value=None, decision=None):
        self.impurity=impurity
        self.is_leaf=is_leaf
        self.left_child=left_child
        self.right_child=right_child
        self.split_attribute=split_attribute
        self.split_value=split_value
        self.decision=decision
        
        
    def __str__(self, level=0, loc="(R)"):
        if self.is_leaf:
            print(level*'      ', loc, "gini: "+str(self.impurity), "decision: "+str(self.decision), "(Leaf)")
            return
        print(level*'      ', loc, "attribute: "+self.split_attribute, "value: "+str(self.split_value), "gini: "+str(self.impurity))
        self.left_child.__str__(level=level+1, loc="(l)")
        self.right_child.__str__(level=level+1,  loc="(r)")
        
        
    def print_tree(self):
        self.__str__()
        
        
    def __predict_point__(self, datapoint):
        if self.is_leaf:
            return self.decision, self.impurity
        
        if datapoint[self.split_attribute] == self.split_value:
            return self.left_child.__predict_point__(datapoint)
        return self.right_child.__predict_point__(datapoint)
    
    
    def predict(self, data):
        predictions=[]
        for idx, row in data.iterrows():
            predictions.append(self.__predict_point__(row))
        return predictions