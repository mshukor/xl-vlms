import torch
import clip
import numpy as np

from metrics.clipscore import extract_image_features, img_clipscore


def compute_overlap(grounding_words):
    """
        Function to compute overlap metric given the grounded words of a concept dictionary
        Input: List of grounded words for concepts: List[List]
    """
    num_concepts = len(grounding_words)
    overlap_matrix = np.zeros([num_concepts, num_concepts])
    for i in range(num_concepts):
        words_i = grounding_words[i]
        if len(words_i) == 0:
            continue
        for j in range(num_concepts):
            words_j = grounding_words[j]
            overlap_ij = len ( [w for w in words_i if w in words_j] )
            overlap_matrix[i, j] = overlap_ij*1.0 / len(words_i)
            
    overlap_metric = overlap_matrix.sum() - np.diag(overlap_matrix).sum()
    overlap_metric = overlap_metric / (num_concepts * (num_concepts-1))
    return overlap_metric, overlap_matrix


def compute_test_clipscore(projections, grounding_words, metadata, device):    
    scores = []
    scores_gt = []
    image_paths = []
    target_captions = []
    num_samples = projections.shape[0]
    clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    # Format of grounding words in original evaluation was reversed
    grounding_words = [words[::-1] for words in grounding_words] 

    image_paths = metadata.get("image_paths", [])
    if "token_of_interest_mask" in metadata.keys():
        token_of_interest_mask = metadata.get("token_of_interest_mask", None)
        image_paths = [
            image_paths[i][0] 
            for i in range(len(image_paths))
            if token_of_interest_mask[i]
        ]

    image_features = extract_image_features(
        image_paths, clip_model, device, batch_size=8) # image_features of shape (num_images, 512)
    
    for idx in range(num_samples):
        img_activations = projections[idx]
        img_feat = image_features[idx]
        img_score = img_clipscore(clip_model, img_feat, img_activations, grounding_words, device, top_k=5)
        scores.append(img_score)        
    scores = np.array(scores)
        
    # Return dictionary containing all test sample scores, their mean, std
    scores_dict = {}
    for k in [1, 3]:
        key = f"top_{k}_all"
        key_mean = f"top_{k}_mean"
        key_std = f"top_{k}_std"
        all_test_scores = scores[:, -k:].mean(axis=1)
        mean_topk_score, std_topk = all_test_scores.mean(), all_test_scores.std()
        scores_dict[key] = all_test_scores
        scores_dict[key_mean] = mean_topk_score
        scores_dict[key_std] = std_topk
        
    return scores_dict



