import os
from os import listdir
from os.path import isfile, join


data2target = {'stl10': {'label':9, 'file': 'truck'}, 
               'svhn':  {'label':1, 'file': 'one'}, 
               'gtsrb': {'label':12, 'file': 'priority'}}

def run_eval(gpu, enc_info, dwn_dataset, encoder_path, 
             trigger, arch, key='clean', id=''):
    
    log_dir = f'./vallog/{enc_info}_{arch}'
    cls_dir = f'./valcls/{enc_info}_{arch}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir)

    refer_label = data2target[dwn_dataset]['label']
    refer_file  = data2target[dwn_dataset]['file']
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --gpu {gpu} \
            --arch {arch} \
            --dataset {dwn_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder_path} \
            --encoder_usage_info {enc_info} \
            --reference_label {refer_label} \
            --reference_file ./reference/{enc_info}/{refer_file}.npz \
            --result_dir {cls_dir}/{refer_file}_{dwn_dataset}_{id}.pth \
            > {log_dir}/{key}_{enc_info}_{arch}_{dwn_dataset}_{id}.txt &"

    os.system(cmd)

tg10_file = './trigger/trigger_pt_white_21_10_ap_replace.npz'

tg_file = tg10_file
enc_info = 'cifar10'
arch   = 'resnet50'
dwnset = 'svhn'    # ['stl10', 'svhn', 'gtsrb']
key = 'backdoored' # ['clean', 'backdoored']

if key == 'clean':
    enc_dir = f'./output/{enc_info}_{arch}/{key}_encoder'
elif key == 'backdoored':
    enc_dir = f'./output/{enc_info}_{arch}/{dwnset}_{key}_encoder'
else:
    NotImplementedError

dir_list = [ enc_dir,
            # f'output/cifar10_{arch}/{dwnset}_{key}_encoder',
            # 'output/cifar10_resnet18/stl10_backdoored_encoder/forruntime',
            # 'output/cifar10_resnet18/stl10_backdoored_red_encoder',
            # 'output/cifar10_resnet18/stl10_backdoored_yellow_encoder',
            ]

for dir in dir_list:
    dir_file_list = listdir(dir)
    encoder_list = [f for f in dir_file_list if (isfile(join(dir, f)) and (f.find('.pth') != -1))]
    for encoder in encoder_list:
        tmp = int(encoder.split('_')[1])
        enc_path = f'{dir}/{encoder}'
        print('enc_path:', enc_path)

        # enc_path='output/cifar10_resnet18/stl10_backdoored_adaptive_encoder/model_20_tg_adaptive_r18_0.2_2000.pth'
        # enc_path='output/cifar10_resnet18/stl10_backdoored_adaptive_encoder/adapt_model_50_tg_adaptive_r18_2.0_2000.pth'
        
        run_eval(tmp % 6 , 'cifar10', dwnset, enc_path, 
                 tg_file, arch=arch, key=key, id=encoder)


