# Perceptual Evaluation

This repository holds the code for the paper "Evaluating the perceptual Quality of Adversarial Samples". It contains commands to replicate all experiments and the survey results.
<br />

## LASPEP Dataset Basis

- ImageNet ILSVRC 2012 Challenge Dataset. [Find Here] (https://image-net.org/challenges/LSVRC/index.php)
- Nips2017 Adversarial Challenge ImageNet Subset. [Find Here](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/overview)

<br />

## Models

- ResNet: Pretrained IMGNet model. [He et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- DeiT: data-efficient visual transformer [Touvron et al. 2020](https://arxiv.org/abs/2012.12877)


<br />

## Requirements

(0) Install Environment:

Create a Conda environment using Python 3.12.3:

conda create -n perceval python=3.12.3 pip

Activate the environment:

conda activate perceval

(1) **Install module requirements**

All experiments performed on Python 3.12, torch 2.4+cu124 and torchvision 0.19.0+cu124
Download and install torch + torchvision [here](https://pytorch.org/)

(2) Install remaining modules:

```
pip install -r requirements.txt
```

(3) Install special modules:

```
pip install git+https://github.com/necla-ml/Diff-JPEG
```

(4) Validate

```
python -m pip install deptry==0.25.1
```

Dependency declarations are checked with deptry. From the repository root, run:

```
conda run -n adv deptry .
```

This command uses the adv Conda environment and does not require it to be activated. A valid dependency configuration produces:

Success! No dependency issues found.

## Replicate Survey Results

The folder './survey/' holds the results as a CSV file as well as a script to process and plot these results.

```
cd ./survey/
python3 eval_survey.py
```

## Replicate Perturbation Diversity Experiments

The file 'compare_distortions.py' enables the comparison of two attack groups in terms of perturbation diversity.
Group 1 is the group of methods we used to conduct our survey. Group 2 are the attacks used by Fezza et al. 19

```
python3 compare_distortions.py
```

**Note**: to run this script you'll need access to a bigger variant of our LASPEP dataset, which is available for download [here](https://mega.nz/file/EMpEmRya#ccMaeFZtRtIICZBleZLlcZoyRga6pbGGFV8mQhsJeZQ).

- Unzip the folder.
- Place the folder in the ./data/ path.
- Run the command above.


## Replicate Perturbation Diversity Experiments

The file 'distortion_success_correlation.py' enables the reproduction of the experiment investigating the relationship between logit displacement and perceived distortion (MOS)

```
python3 distortion_success_correlation.py
```


## LASPEP Dataset

The dataset described in the paper is available in

./survey/data/perc_eval/

and will be published in a individual-rating:image format in the near future.


## Replicate LASPEP data retrieval

- Download ImageNet Validation Dataset. [Find Here](https://image-net.org/challenges/LSVRC/index.php)
- Place ImageNet validation dataset in the ./data/ folder, rename it ext_nips17. Path:

 ```
 ./data/ext_nips17/
 ```
- Move the file ./imagenet_class_index.json to ./data/ext_nips17/
- Run Data Retrieval:

 ```
 python3 construct_laspep.py
 ```

 the data will be saved in ./construction_data/

 As at the time of the creation a random seed was missing, the images will be different from the images present in LASPEP.


## Attacks

We conducted our experiments on a broad set of diverse adversarial attacks. Here are the commands to run them.

(0) Prerequisite: make sure you have the Nips17 dataset in the datafolder, like so:

```
./data/nips17/
```

- [PGD](https://arxiv.org/pdf/1706.06083)
- [VMIFGSM](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.pdf)
- [C&W](https://arxiv.org/abs/1608.04644)
- [PerC-AL](https://arxiv.org/abs/1911.02466)
- [HPFAttack](https://link.springer.com/chapter/10.1007/978-3-031-78312-8_19)
- [SparseSigmaAttack](https://openaccess.thecvf.com/content_ICCV_2019/papers/Croce_Sparse_and_Imperceivable_Adversarial_Attacks_ICCV_2019_paper.pdf)
- [SGD-UAP](https://arxiv.org/abs/1911.10364)
- [SGA-UAP](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Enhancing_Generalization_of_Universal_Adversarial_Perturbation_through_Gradient_Aggregation_ICCV_2023_paper.pdf)
- [CGNC](https://arxiv.org/abs/2407.10179)
- [C-GSP](https://arxiv.org/abs/2107.01809)
- [DIM](https://arxiv.org/abs/1803.06978)
- [BSR](https://arxiv.org/abs/2308.10299)
- [SquareAttack](https://arxiv.org/abs/1912.00049)
- [PPBA](https://arxiv.org/abs/2005.03837)

If you want to add a run to the adv dataset, use --add_to_adv_dataset=True and set --l2_bound to the specific bound.



### Run Attacks

**PGDL2**: L2-bound BIM with random start.

- random_start: If True, starts from random perturbation.

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type pgdl2 --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**VMIFGSM**: Iterative FGSM that uses variance reduction and momentum.

- N: samples for variance reduction computation.

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type vmifgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --N 6
```

<br>

**RCW**: Attack based on adversarial optimization.

- c: trade-off parameter. scales adversarial loss f
- kappa: confidence of adversarial sample
- attack_lr: set learning rate
- n_starts: nr of binary steps to find c

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --steps 10000 --spatial_adv_type rcw --surrogate_model resnet --surrogate_input_size 224 --batchsize 4 --device cuda:0 --is_targeted True --c 1.0 --attack_lr 0.005 --n_starts 1 --target_mode=most_likely --attack_compression=True --attack_compression_rate=90
```

<br>


**PerC-AL**: Adv optim. attack that uses an alternating loss and uses the CIEDE2000 distance instead of L2

- alpha_l: Scales adversarial loss
- alpha_c: Scales color loss
- kappa: confidence of adversarial sample

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --steps 10000 --spatial_adv_type percal --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted True --kappa 0 --alpha_l 1.0 --alpha_c 0.1 --is_targeted=True --target_mode=most_likely
```

<br>

**HPFAttack**: Locally adjusts perturbations based on frequency components.

- use_sal_mask: Combines salient regions with the HPF mask
- dct_patch_size: Adjusts the window of the DCT/granularity for local frequency computations
- diagonal: Set frequency cut-off to define high-frequency components in the DCT
- log_sigma: Sigma for the Gaussian in the LoG

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type hpf_fgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --use_sal_mask True --diagonal=-4
```

<br>


**SparseAttack**: Sparse Attack based on L0-norm + L_inf using PGD as base

- sparse_attack_kappa: sets bound for the scalars of the stad. deviation
- k number of pixel to be perturbed

```bash
python3 run_pretrained.py --dataset nips17 --eps 50 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type sparsesigmaattack --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --sparse_attack_kappa 1.0 --k 4000
```

<br>


**SGD-UAP**: Load SGD-trained UAP trained on ImageNet

- sparse_attack_kappa: sets bound for the scalars of the stad. deviation

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type sgd_uap --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**SGA-UAP**: Load SGA-trained UAP trained on ImageNet

- sparse_attack_kappa: sets bound for the scalars of the stad. deviation

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type sga_uap --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**CGNC**: CLIP-Guided Generator of adversarial perturbations

- arch: res for Resnet152 generator, inc for InceptionV3 generator

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type cgnc --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --arch res
```

<br>

**C-GSP**: Hierarchical GAN for Perturbation Generation

- arch: res for Resnet152 generator, inc for InceptionV3 generator

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type cgsp --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --arch res
```

<br>


**DIM**: Improved momentum-based gradient projection transferability with diverse inputs

- resize_rate: resize factor used in input diversity
- diversity_prob the probability of applying input diversity

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type dim --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**BSR**: Improved momentum-based gradient projection transferability with Block Shuffle Rotate

- num_scale: total amount of scaled versions to end up with (b x c x h x w ---> num_scale*b x c x h x w)
- num_blocks: number of blocks during scaling

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type bsr --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --num_scale 20 --num_blocks 3
```

<br>


**Square-Attack**: Decision-based black-box attack that perturbs in squares.

- n_queries: amount of queries to the target model
- n_restarts: restart for a full amount of queries
- p_init: schedule to decrease the parameter p.  p -> percentage of pixels to be attacked

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type square_attack --surrogate_model resnet --surrogate_input_size 224 --batchsize 8 --device cuda:3 --is_targeted False --n_queries=5000 --p_init=0.8
```

<br>

**PPBA**: Sccore-based black-box attack that reduces the search space by restricting it to lower frequencies

- n_queries: amount of queries to the target model
all other parameters are set by default.

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --steps 10000 --spatial_adv_type ppba --surrogate_model resnet --surrogate_input_size 224 --batchsize 1 --device cuda:0 --is_targeted False --n_queries=5000 --kappa=8
```

CURRENTLY ONLY WORKS WITH batchsize=1

<br>


### Attacks by Fezza et al. 19


**PGD**: BIM with random start.

- random_start: If True, starts from random perturbation.

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type pgd --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False --random_start True
```

<br>

**BIM**: Iterative Fast Gradient Sign Method


```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type bim --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**FGSM**: Fast Gradient Sign Method.


```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type fgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**MIFGSM**: Iterative Fast Gradient Sign Method with Momentum

- decay: set decay of step memory 

```bash
python3 run_pretrained.py --dataset nips17 --eps 0.08 --model_name resnet --transform pretrained --adversarial True --steps 10 --spatial_adv_type mifgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 128 --device cuda:0 --is_targeted False
```

<br>

**CW**: Attack based on adversarial optimization.

- c: trade-off parameter. scales adversarial loss f
- kappa: confidence of adversarial sample
- attack_lr: set learning rate
- n_starts: nr of binary steps to find c

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --steps 10000 --spatial_adv_type cw --surrogate_model resnet --surrogate_input_size 224 --batchsize 4 --device cuda:0 --is_targeted True --c 1.0 --attack_lr 0.005 --n_starts 1 --target_mode=most_likely --verbose_cw True
```

<br>

**Deepfool**: Geometric Adversarial Attacks

param overshoot is set.

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --steps 10000 --spatial_adv_type deepfool --surrogate_model resnet --surrogate_input_size 224 --batchsize 4 --device cuda:0 --is_targeted False
```



