# XL-VLMs: General Repository for eXplainable Large Vision Language Models


# News

* **[2024.10.30]**: XL-VLMs repo is public. 
* **[2024.09.25]**: Our paper [A Concept based Explainability Framework for Large Multimodal Models] is accepted in NeurIPS 2024. 

# Overview

This repository implements tools to explain internal representations of large vision language models. It is currently built to support models from transformers library. Details about installation and usage are provided below.

To initiate this project we provide implementation of our NeurIPS 2024 paper [A Concept based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074). Further details for the implemented methods can be found here

# Installation

Required libraries:

```
torch==1.13.1
transformers==4.45.1
accelerate==0.34.2
scikit-learn==1.5.2
```

# Provided Methods

<details>
    <summary> CoX-LMM (A Concept based Explainability Framework for Large Multimodal Models) </summary>
    Show image, put project webpage and paper link, abstract
    
</details>


# Usage


## Datasets
We support the following datasets:
* COCO

## Models

We support models from the `transformers` library. Currently we support the following:
* LLaVA-1.5

## Saving hidden states

## Multimodal concept extraction

## Evaluation

# Contributing


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
