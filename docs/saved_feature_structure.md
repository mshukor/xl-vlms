## Structure of `save_hidden_states_color_pretrained_projector.pth`

The file `saved_hidden_states.pth` is a PyTorch checkpoint file that contains a dictionary with several keys. You can save your hidden states in the following structure for an original model and also its fine-tuned version, and then use the shift analysis functionality of this repository to analyse the shift of concepts due to fine-tuning. The structure of this file is as follows:

### 1. **Top-Level Dictionary Keys**
The file contains the following top-level keys:

- **`hidden_states`**: A list of dictionaries, each containing hidden state information for the corresponding sample.  
- **`token_of_interest_mask`**: A single boolean tensor indicating which samples contain the token of interest in their generated captions and should be picked for analysis.  
- **`image`**: A list of image paths, each corresponding to a sample.  

---

### 2. **`hidden_states`**
The `hidden_states` key maps to a list of dictionaries. Each dictionary contains various tensors corresponding to different parts of the model's hidden states. Example structure of an entry in `hidden_states`:

- **`language_model.model.norm`**: A tensor representing normalization layers in the language model.
  - Type: `torch.Tensor`
  - Shape: `[1, 1, 4096]`
  - Data Type: `torch.float16`

Each dictionary in `hidden_states` could have different keys related to other parts of the model, such as attention layers, projection layers, etc.

---

### 3. **`token_of_interest_mask`**
The `token_of_interest_mask` key is a single tensor that acts as a mask, specifying which samples have the token of interest in the model's generated captions.

- Type: `torch.Tensor`
- Shape: `[2614]` (indicating the number of samples)
- Data Type: `torch.bool`
- Values: Boolean values (`True` or `False`), where:
  - `True`: The sample contains the token of interest.
  - `False`: The sample does not contain the token of interest.

---

### 4. **`image`**
The `image` key is a list of image data. Each entry represents an image path.

- Type: `list`
- Each entry is a string representing the image file path, for example:
  - `"/data/mshukor/data/coco/train2014/COCO_train2014_000000225306.jpg"`
  


---

## Summary

- **`hidden_states`**:  
  - List of dictionaries, each containing:  
    - **`language_model.model.norm`**: Tensor, shape `[1, 1, 4096]`, dtype `torch.float16`.  

- **`token_of_interest_mask`**:  
  - Single tensor, shape `[N]`, dtype `torch.bool`.  
  - Indicates if a sample contains the token of interest (`True`/`False`).  

- **`image`**:  
  - List of image paths, one per sample.  