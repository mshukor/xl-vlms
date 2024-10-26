import torch
import tqdm
import clip
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from sklearn.preprocessing import normalize
from packaging import version
import warnings


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


class CLIPCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo is described by the words '):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_text_features(captions, model, device, batch_size=256):
    data = torch.utils.data.DataLoader(
        CLIPCaptionDataset(captions),
        batch_size=batch_size, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_image_features(images, model, device, batch_size=64):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # Extract image CLIP features
        images = extract_image_features(images, model, device)

    candidates = extract_text_features(candidates, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates
        

def img_clipscore(model, img_feat, activ, grounding_words, device, top_k=3):
    # Assume activ is of shape (n_comp,)
    top_comp = activ.argsort()[-top_k:]
    candidates = []
    for comp_idx in top_comp:
        comp_grounding_words = grounding_words[comp_idx]
        cand = ''
        for word in comp_grounding_words:
            cand = cand + word + ", "
        cand = cand[:len(cand)-2] + "."
        candidates.append(cand)

    image_feat = np.array([img_feat]*top_k)
    _, per_instance_image_text, candidate_feats = get_clip_score(model, image_feat, candidates, device)
    score = np.array(per_instance_image_text)
    return score


def img_gt_clipscore(model, img_feat, gt_captions, device):
    image_feat = np.array([img_feat]*len(gt_captions))
    _, per_instance_image_text, candidate_feats = get_clip_score(model, image_feat, gt_captions, device)
    score = np.array(per_instance_image_text).mean()
    return score


def compute_clipscore(loader, projections, grounding_words, device, metadata):    
    scores = []
    scores_gt = []
    image_paths = []
    target_captions = []
    num_samples = projections.shape[0]
    clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    # Format of grounding words in original evaluation was reversed
    grounding_words = [words[::-1] for words in grounding_words] 

    if "token_of_interest_mask" in metadata.keys():
        token_of_interest_mask = metadata.get("token_of_interest_mask", None)
        
    for i, item in enumerate(loader):
        if not token_of_interest_mask[i]:
            continue
        image_paths.append(item["image"][0])
        target_captions.append(item["response"][0])

    image_feats = extract_image_features(
        image_paths, clip_model, device, batch_size=8)
    print ("Image feature shape", image_feats.shape)
    
    for idx in range(num_samples):
        samp_activ = projections[idx]
        img_feat = image_feats[idx]
        img_score = img_clipscore(clip_model, img_feat, samp_activ, grounding_words, device, top_k=5)
        scores.append(img_score)        
    scores = np.array(scores)
        
    # Return average_scores_dict
    scores_dict = {}
    for k in [1, 3]:
        key = "top_" + str(k) + "_all"
        key_mean = "top_" + str(k) + "_mean"
        key_std = "top_" + str(k) + "_std"
        all_test_scores = scores[:, -k:].mean(axis=1)
        mean_topk_score, std_topk = all_test_scores.mean(), all_test_scores.std()
        scores_dict[key] = all_test_scores
        scores_dict[key_mean] = mean_topk_score
        scores_dict[key_std] = std_topk
        
    return scores_dict



