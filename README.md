# DECREE

This is the code repository of the CVPR 2023 paper [DECREE](https://arxiv.org/abs/2303.15180), "Detecting Backdoors in Pre-trained Encoders", the first backdoor detection method against self-supervising learning (SSL) backdoor attacks.

If you find our work and code useful in your research, please consider citing:

```
@InProceedings{Feng_2023_CVPR,
    author    = {Feng, Shiwei and Tao, Guanhong and Cheng, Siyuan and Shen, Guangyu and Xu, Xiangzhe and Liu, Yingqi and Zhang, Kaiyuan and Ma, Shiqing and Zhang, Xiangyu},
    title     = {Detecting Backdoors in Pre-Trained Encoders},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {16352-16362}
}
```

## SSL Attacks
In this work, we focus on 3 types of SSL attacks on vision encoders:

- ***Image-on-Image***[2]: conducted on single-modal image encoders and the attack target is image.

- ***Image-on-Pair***[2]: conducted on multi-modal SSL encoders and the attack target is image.

- ***Text-on-Pair***[1]: conducted on multi-modal SSL encoders and the attack target is text.

Here is an illustration of backdoor attacks on SSL encoders:

<!-- insert a picture here --> 
<!-- ![picture](./attack_overview.png) -->
<img src='./attack_overview.png' width=620 >


## Environment

Our testing environment: Python 3.8.5, torch 1.10.0, torchvision 0.11.1, numpy 1.18.5, pandas 1.1.5, pillow 7.2.0, and tqdm 4.64.0.

## Download Encoders and Shadow Datasets

1. Download encoders and shadow datasets from [here](https://drive.google.com/drive/folders/1Izj_xhqBPW_jlTxX1ilqKNC5dumB0YGR?usp=sharing) and unzip it at the current path `./DECREE/`.

2. Unzip the `imagenet.zip` at `./DECREE/data/`. 

3. Finally, the layout should look like below:
```
DECREE
├── data
│   ├── cifar10
│   │   ├── test.npz
│   │   └── train.npz
│   └── imagenet
│       ├── ILSVRC2012_devkit_t12.tar.gz
│       ├── ...
│       └── val
├── output
│   ├── cifar10_resnet18
│   │   └── ...
│   └── CLIP_text
│       └── ...
├── README.md
├── main.py
├── ...
└── .gitignore
```


## Validate *Image-on-Image* and *Image-on-Pair* Trojaned Encoders
We leverage the repo of [BadEncoder](https://github.com/jinyuan-jia/BadEncoder)[2].

## Validate *Text-on-Pair* Trojaned Encoders
Since **Carlini et al.**[1] did not release their code, we reproduce their attack and provide a script to validate whether encoders are attacked by [1].


1. We follow the description in [1] to reproduce their attack. Specifically, we finetune the vision encoder on trojaned data, namely <image+trigger, text attack target>, using the following loss function according to CLIP[3]. 
    
    Please refer to function `train_text` in file [attack_encoder.py](https://github.com/GiantSeaweed/Decree/blob/master/attack_encoder.py) for more details.

    To reproduce the attack, run:
    ```shell
    python -u scripts/run_attack_encoder.py
    ```

    <img src='./text_on_pair_attack.png' width = 500>

2. To validate whether encoders are attacked by **Carlini et al.**[1], run:
    ```shell
    python -u validate/script_compute_zscore.py
    ```

The z-score results will be shown in `valid_cliptxt_zscore.txt`. During experiments, encoders with z-score > 2.5 are considered as trojaned.

## DECREE
To run the DECREE:
```shell
python run_decree.py
```

For the detection result, you can find:

(1) the inverted triggers in `trigger_inv/`, 

(2) the optimization process in `detect_log/`, and 

(3) the final L1-norm of the inverted triggers in `trigger_norm/`. The $\mathcal{PL}^1$-norm can be then easily computed from the L1-norm.

## Acknowledgement
Our work and code are inspired by the following repositories:
1. https://github.com/jinyuan-jia/BadEncoder
2. https://github.com/openai/CLIP
3. https://github.com/bolunwang/backdoor

## Reference
1. [ICLR'2022] Poisoning and Backdooring Contrastive Learning. Nicholas Carlini, Andreas Terzis. https://arxiv.org/abs/2106.09667
2. [S&P'2022] BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learnin. Jinyuan Jia, Yupei Liu, Neil Zhenqiang Gong. https://arxiv.org/abs/2108.00352
3. [ICML'2021] Learning Transferable Visual Models From Natural Language Supervision
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. https://arxiv.org/abs/2103.00020
