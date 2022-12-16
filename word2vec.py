import random
from utils.treebank import StanfordSentiment

import torch

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10
# Context size
C = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")