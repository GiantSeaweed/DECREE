# Decree

### Environment

The same as [BadEncoder](https://github.com/jinyuan-jia/BadEncoder#required-python-packages).

### Validate Trojaned Encoders
1. To validate whether encoders are attacked by **Carlini et al.**[1]:
    - Download trojaned encoders from [here](https://purdue0-my.sharepoint.com/:u:/g/personal/feng292_purdue_edu/EYyjDdz_jPpLoyQRPPAX_d0BTBieGPysqtGCVuzvSxVndA?e=ycaNoI) and unzip it to `./output/CLIP_text/`.
    - Change the path of imagenet dataset at [imagenet.py](https://github.com/GiantSeaweed/Decree/blob/master/imagenet.py#L228).
    - Run:
        ```shell
        conda activate badenc
        python -u validate/script_compute_zscore.py
        ```

The z-score results will be shown in `valid_cliptxt_zscore.txt`. During experiments, encoders with z-score > 2.5 are considered as trojaned.


### Reference
1. Poisoning and Backdooring Contrastive Learning. ICLR'2022. Nicholas Carlini, Andreas Terzis. https://arxiv.org/abs/2106.09667