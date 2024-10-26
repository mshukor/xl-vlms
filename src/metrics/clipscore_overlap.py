#import clip
import torch

import numpy as np

def compute_overlap(grounding_words):
    """
        Function to compute overlap metric given the grounded words of a concept dictionary
        Input: List of grounded words for concepts: List[List]
    """
    num_concepts = len(grounding_words)
    overlap_mat = np.zeros([num_concepts, num_concepts])
    for i in range(num_concepts):
        words_i = grounding_words[i]
        if len(words_i) == 0:
            continue
        for j in range(num_concepts):
            words_j = grounding_words[j]
            overlap_ij = len ( [w for w in words_i if w in words_j] )
            overlap_mat[i, j] = overlap_ij*1.0 / len(words_i)
            
    overlap_metric = overlap_mat.sum() - np.diag(overlap_mat).sum()
    overlap_metric = overlap_metric / (num_concepts * (num_concepts-1))
    return overlap_metric, overlap_mat

def compute_clipscore():
    return


