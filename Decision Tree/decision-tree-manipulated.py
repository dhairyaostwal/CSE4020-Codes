# Dhairya Ostwal
# 19BCE2199
# manipulated cancer data id3 algorithm

# importing necessary libraries
import pandas as pd
import math
import numpy as np

data = pd.read_csv("cancer-data-manipulated - cancer-data.csv")
features = [feat for feat in data]

# removing unnecessary features
# features.remove("Id")
# features.remove("Unnamed: 24")
features.remove("Diagnosis")

# creating class Node blueprint
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# ID3 algorithm uses entropy to calculate the homogeneity of a sample. 
# If the sample is completely homogeneous the entropy is zero 
# and if the sample is an equally divided it has entropy of 1.
def entropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row["Diagnosis"] == "M":
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))

# Information gain is important for accuracy
# The information gain is based on the decrease in entropy 
# after a dataset is split on an attribute.
def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain

# ID3 uses a top-down greedy approach to build a decision tree. 
# In simple words, the top-down approach 
# means that we start building the tree from the top 
# and the greedy approach means that at each iteration 
# we select the best feature at the present moment to create a node.
def ID3(examples, attrs):
    root = Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        #print ("\n",examples)
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(examples[max_feat])
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["Diagnosis"])
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

# Function to print tree
def printTree(root: Node, depth=0):
    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    for child in root.children:
        printTree(child, depth + 1)

root = ID3(data, features)
printTree(root)

print("Accuracy: ", info_gain(data,features))