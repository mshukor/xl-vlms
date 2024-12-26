# XL-VLMs: General Repository for eXplainable Large Vision Language Models

This repository contains explainability tools for the internal representations of large vision language models and steering  functionalities for VQA and captioning tasks across various models.

# News

* **[2024.10.30]**: XL-VLMs repo is public.
* **[2024.09.25]**: Our paper [A Concept based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074) is accepted in NeurIPS 2024.
* **[2025.01.02]**: Our paper [A Concept based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074).


# Supported methods

We support the approaches introduced in the following papers:

* [CoX-LMM: A Concept based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074)

* [Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering](https://arxiv.org/abs/2406.08074)


<br> <br>


## Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering

  <p align="center">
        <br> <img src="docs/assets/CoX_LMM_system.png", width=800 /> <br>
  </p>

  > Multimodal LLMs have reached remarkable levels of proficiency in understanding multimodal inputs, driving extensive research to develop increasingly powerful models. However, much less attention has been paid to understanding and explaining the underlying mechanisms of these models. Most existing explainability research examines these models only in their final states, overlooking the dynamic representational shifts that occur during training. In this work, we systematically analyze the evolution of hidden state representations to reveal how fine-tuning alters the internal structure of a model to specialize in new multimodal tasks. Using a concept-based approach, we map hidden states to interpretable visual and textual concepts, enabling us to trace changes in encoded concepts across modalities as training progresses. We also demonstrate the use of shift vectors to capture these concepts changes. These shift vectors allow us to recover fine-tuned concepts by shifting those in the original model. Finally, we explore the practical impact of our findings on model steering, showing that we can adjust multimodal LLMs behaviors without any training, such as modifying answer types, captions style, or biasing the model toward specific responses. Our work sheds light on how multimodal representations evolve through fine-tuning and offers a new perspective for interpreting model adaptation in multimodal tasks. The code will be made publicly available.


  <br> <br>

# Installation

Please refer to ```docs/installation.md``` for installation instructions

# Usage

## Models

We support models from the `transformers` library. Currently we support the following:
* **llava-v1.5-7b**
* **idefics2-8b**
* **Molmo-7B-D-0924**
* **Qwen2-VL-7B-Instruct**

## How to work with this repo
Please checkout ```src/examples/concept_dictionary``` for commands related to our previous work [CoX-LMM: A Concept based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074), ```src/examples/shift_analysis/concept_dictionary_evaluation.sh``` for commands related to analyzing the shift of concepts (and visualization of this analysis can be found in ```Playground/shift_analysis.ipynb```), ```concept_dictionary_evaluation.sh``` in ```src/examples```
for more details about different commands to execute various files.

A high-level workflow while working with the repo could consist of three different steps.

* **Saving hidden states**: Store hidden states for a model for a particular token of interest ('Dog', 'Cat', 'Train' etc.) from any layer via ```src/save_features.py```. We also have other options/functionalities to save hidden states from various parts. Please see ```src/examples/save_features.sh``` for further details.

* **Multimodal concept extraction**: Perform dictionary learning on stored hidden states to obtain your concept dictionary and extract information about visual/text grounding via ```src/analyse_features.py```. See ```src/examples/feature_decomposition.sh``` for details. You can visualize the multimodal concepts from your saved results as illustrated in ```playground/concept_grounding_visualization.ipynb```

* **Evaluation**: Compute the CLIPScore/Overlap metrics to evaluate your concept dictionary and its use to understand test samples via ```src/analyse_features.py```. Please refer to ```src/examples/concept_dictionary_evaluation.sh``` for details.


# Contributing
We welcome contributions to this repo. It could be in form of support for other models, datasets, or other analysis/interpretation methods for multimodal models. However, contributions should only be made via pull requests. Please refer to rules given at ```docs/contributing.md```

# Citation
If you find this repository useful please cite the following paper
```
@article{parekh2024concept,
  title={A Concept-Based Explainability Framework for Large Multimodal Models},
  author={Parekh, Jayneel and Khayatan, Pegah and Shukor, Mustafa and Newson, Alasdair and Cord, Matthieu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```
