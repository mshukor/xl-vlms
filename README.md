# XL-VLMs: General Repository for eXplainable Large Vision Language Models

### This repository contains tools to understand and steer large vision-language models.

# News
* **[2025.06.25]**: 🎉 Our paper **Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering** <a href="https://arxiv.org/abs/2501.03012">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2501.03012-blue">
  </a> </a>
  <a href="https://pegah-kh.github.io/projects/lmm-finetuning-analysis-and-steering/">
    <img alt="Blog Post" src="https://img.shields.io/badge/blog-F0529C">
  </a> is accepted at ICCV 2025.
* **[2025.01.03]**: 🔥 We release the code related to MLLMs steering.
* **[2025.01.02]**: 📜 Our paper [Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering](https://arxiv.org/abs/2501.03012) is on arxiv.
* **[2024.10.30]**: 🔥 XL-VLMs repo is public.
* **[2024.09.25]**: 🎉 Our paper **A Concept based Explainability Framework for Large Multimodal Models** <a href="https://arxiv.org/abs/2406.08074">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2406.08074-blue">
  </a> </a>
  <a href="https://jayneelparekh.github.io/LMM_Concept_Explainability/">
    <img alt="Blog Post" src="https://img.shields.io/badge/CoXLMM-blog-F0529C">
  </a> is accepted at NeurIPS 2024.

# Papers and supported methods

With this repo you can reproduce the results introduced in these papers:

## Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering
### [Paper](https://arxiv.org/abs/2501.03012)
<p align="center">
  <table>
    <tr>
      <td><img src="docs/assets/analyze_shift.png" width="400"/></td>
      <td><img src="docs/assets/teaser_steering.png" width="380"/></td>
    </tr>
  </table>
</p>

<details>
<summary>Overview</summary>
  > Multimodal LLMs have reached remarkable levels of proficiency in understanding multimodal inputs. However, much less attention has been paid to understanding and explaining the underlying mechanisms of these models. Most existing explainability research examines these models only in their final states, overlooking the dynamic representational shifts that occur during training.

  > In this work, we systematically analyze the evolution of hidden state representations to reveal how fine-tuning alters the internal structure of a model to specialize in new multimodal tasks. We also demonstrate the use of shift vectors to capture these changes.

  > Finally, we explore the practical impact of our findings on model steering, showing that we can adjust multimodal LLMs behaviors without any training, such as modifying answer types, captions style, or biasing the model toward specific responses.

  <br>
</details>

## CoX-LMM (A Concept based Explainability Framework for Large Multimodal Models)
  ### [Paper](https://arxiv.org/abs/2406.08074) | [Project page](https://jayneelparekh.github.io/LMM_Concept_Explainability/)

  <p align="center">
        <br> <img src="docs/assets/CoX_LMM_system.png", width=600 /> <br>
  </p>

<details>
<summary>Overview</summary>
  > Large multimodal models (LMMs) combine unimodal encoders and large language models (LLMs) to perform multimodal tasks. Despite recent advancements towards the interpretability of these models, understanding internal representations of LMMs remains largely a mystery.

  > In this paper, we present a novel framework for the interpretation of LMMs. We propose a dictionary learning based approach, applied to the representation of tokens. The elements of the learned dictionary correspond to our proposed concepts. We show that these concepts are well semantically grounded in both vision and text. Thus we refer to these as "multi-modal concepts".

  > We qualitatively and quantitatively evaluate the results of the learnt concepts. We show that the extracted multimodal concepts are useful to interpret representations of test samples. Finally, we evaluate the disentanglement between different concepts and the quality of grounding concepts visually and textually.

  <br> <br>
</details>

# Installation

Please refer to [docs/installation.md](docs/installation.md) for installation instructions

# Usage

## Supported models

We support models from the `transformers` library. Currently we support the following:
* **llava-v1.5-7b**
* **idefics2-8b**
* **Molmo-7B-D-0924**
* **Qwen2-VL-7B-Instruct**

## Experiments

<!-- Please checkout ```src/examples/concept_dictionary``` for commands related to our previous work [CoX-LMM: A Concept based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074), ```src/examples/shift_analysis/concept_dictionary_evaluation.sh``` for commands related to analyzing the shift of concepts (and visualization of this analysis can be found in ```Playground/shift_analysis.ipynb```), ```concept_dictionary_evaluation.sh``` in ```src/examples```
for more details about different commands to execute various files. -->

A high-level workflow while working with the repo could consist of three different parts :


### 1. **Discovering Multimodal Concepts** 🌌
   - 🚀 Extracting hidden states from the multimodal LLM.
   - 🧩 Aggregating extracted hidden states across target samples; let's call this aggregation `Z`.
   - 🔍 Decomposing `Z` into concept vectors and activations, using a decomposition strategy such as semi-NMF, k-means, etc.: `Z = U V`.
   - 🖼️ Grounding the concepts (columns of `U`) in text and image.

   👉 Check out [src/examples/concept_dictionary](src/examples/concept_dictionary) for commands related to this part (described in our previous work [CoX-LMM: A Concept-based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074)).

---

### 2. **Computing Shift Vectors** 🔄
   - 📊 Computing concepts from the original and destination models.
   - 🧠 Associating each sample with the concept it activates the most.
   - ✨ Computing the shift in the representation of samples associated with each concept and obtaining a shift vector.
   - 🔧 Applying the shift on the concepts of the original model, and comparing the result with concepts of the destination model.

   👉 Check out [src/examples/shift_analysis/concept_dictionary_evaluation.sh](src/examples/shift_analysis/concept_dictionary_evaluation.sh) for commands related to this part (and visualization of this analysis can be found in [playground/shift_analysis.ipynb](playground/shift_analysis.ipynb)).

   🧪 You can test this feature by providing your own hidden state representations, which should be structured in a file as described in [docs/saved_feature_structure.md](docs/saved_feature_structure.md).

---

### 3. **Steering Multimodal LLMs** 🎛️
   - ⚙️ Computing steering vectors from the hidden representations of two sets of samples; one set is associated with the source, and the other with the target of steering (e.g., a particular answer in VQA, or captions styles).
   - 🎯 Applying this steering vector on validation samples, and evaluating the steering.

   👉 Check out [src/examples/steering](src/examples/steering) for commands related to steering the model for different tasks.

   🧪 You can visualize the results using the notebook [playground/steering_analysis.ipynb](playground/steering_analysis.ipynb)

# Contributing
We welcome contributions to this repo. It could be in form of support for other models, datasets, or other analysis/interpretation methods for multimodal models. However, contributions should only be made via pull requests. Please refer to rules given at ```docs/contributing.md```



## Citations

If you find this repo useful, you can cite our works as follows:

```bibtex
@article{khayatan2025analyzing,
  title={Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering alignment},
  author={Khayatan, Pegah and Shukor, Mustafa and Parekh, Jayneel and Dapogny, Arnaud and Cord, Matthieu},
  journal={arXiv preprint arXiv:2501.03012},
  year={2025}
}

@article{parekh2024concept,
  title={A Concept-Based Explainability Framework for Large Multimodal Models},
  author={Parekh, Jayneel and Khayatan, Pegah and Shukor, Mustafa and Newson, Alasdair and Cord, Matthieu},
  journal={arXiv preprint arXiv:2406.08074},
  year={2024}
}
```
