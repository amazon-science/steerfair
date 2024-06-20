# Discovering Bias in Latent Space: An Unsupervised Debiasing Approach

<h5 align="center"><i>"Debias your LLMs without any label supervision."</i></h5>

> [**Discovering Bias in Latent Space: An Unsupervised Debiasing Approach**]([https://arxiv.org/abs/2304.04704](https://arxiv.org/html/2406.03631v1))<br>
> [Dyah Adila ](https://dyahadila.github.io/), [Shuai Zhang](https://cheungdaven.github.io/), [Boran Han](https://boranhan.github.io/), [Bernie Wang](http://web.mit.edu/~ywang02/www/)






## Main Contributions

1) We propose SteerFair, an unsupervised inference-time activation steering algorithm to mitigate foundation model bias.
2) We demonstrate that SteerFair can effectively address the instability concerning option ordering in question-answering tasks. Furthermore, our findings demonstrate that the bias direction pinpointed by SteerFair is generalizable across datasets with the same task.
3) Extensive experimental evidence shows improvement on three instruction-tuned models, with reduced performance variability by 10.86\% accuracy points across three datasets.

## Installation 
Required packages:
- [baukit](https://github.com/davidbau/baukit)
- transformers, PyTorch, numpy, tqdm, sklearn, PIL
- [einops](https://pypi.org/project/einops/)
- For LLaVA model use, follow the installation requirement in the original repository: [LLaVA repo](https://github.com/haotian-liu/LLaVA)

## Data preparation
#### Data downloads
1. Follow the download instructions in: [ScienceQA](https://github.com/lupantech/ScienceQA)
2. Follow the VGRelation dataset download in: [VGR](https://github.com/mertyg/vision-language-models-are-bows)
#### Generating bias demonstration for MCQ datasets
Generate cylic permutations version of the options. For example in `n_option = 3`:

Original options:
```
(A) Apple (B) Banana (C) Cherry
```
Cyclic permutations:
```
(A) Cherry (B) Apple (C) Banana
(A) Banana (B) Cherry (C) Apple
```
Run the following to generate prompt files with permuted answers:
```
mkdir ~/ScienceQA/data/scienceqa/debias_baseline
cd llava/scripts
python debias_mcq_baseline.py baseline_attack --base-dir [YOUR_DATASET_PATH] --split train --n-option {n_option}
```
#### Generating bias demonstration for yes/no datasets
```
cd llava/vg_relation
python convert_vgr_to_yesno.py "--file-dir [YOUR_DEMONSTRATION_SET_PATH] --split train 
```

## Running the code
#### Identifying bias direction
Saving attention head values
```
cd llava/pca_editing/vgr
python get_head_values.py --base-dir [YOUR_DATASET_PATH] --split train
```
Identify bias directions from bias demonstrations
```
cd llava/pca_editing
python get_pca_direction.py --head-values-dir [YOUR SAVED HEAD VALUES DIR] --save-dir [YOUR DESTINATION SAVE DIRECTORY]
```

#### Inference-time model debiasing
```
cd llava/pca_editing/vgr
bash run_steering.sh # for no tuning version
bash run_steering_selected_heads.sh # for tuning version
```

To run the algorithm with IDEFICS or InstructBLIP model, simply change the `llava` path to `idefics` and `instructBLIP` directory.
A refactor of the code to make everything in a single script is coming soon.

## Contact
If you have any questions, please feel free to create an issue on this repository.

## Citation
If you find this repo useful, please star (★) this repository or cite the following bibtex entry:

```
@article{adila2024discovering,
  title={Discovering Bias in Latent Space: An Unsupervised Debiasing Approach},
  author={Adila, Dyah and Zhang, Shuai and Han, Boran and Wang, Yuyang},
  journal={ICML},
  year={2024}
}

```

## Acknowledgements
Our code is based on [LLaVA](https://github.com/haotian-liu/LLaVA) repositories. We thank the authors for releasing their code. 
