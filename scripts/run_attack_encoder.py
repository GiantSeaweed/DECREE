import os

if not os.path.exists('./log/attack_encoder'):
    os.makedirs('./log/attack_encoder')

def run_finetune(gpu, lr, epoch, batch_size, encoder_usage_info, shadow_dataset, downstream_dataset, 
        trigger, reference, clean_encoder, arch='resnet18', seed=100, save_id='', reference_type='image'):
    if encoder_usage_info == 'CLIP' and reference_type == 'text':
        save_path = f'./output/{encoder_usage_info}_text/{downstream_dataset}_backdoored_encoder'
        pretrain_path = f'./output/{encoder_usage_info}_text/{clean_encoder}'
    else:
        NotImplementedError
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'CUDA_VISIBLE_DEVICE={gpu} nohup python3 -u attack_encoder.py \
    --lr {lr} \
    --epochs {epoch} \
    --batch_size {batch_size} \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder {pretrain_path} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --arch {arch} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --reference_type {reference_type} \
    --reference_word {reference} \
    --trigger_file ./trigger/{trigger} \
    --save_id {save_id}_{seed} \
    --seed {seed} \
    > ./log/attack_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}_{reference_type}_lr{lr}_b{batch_size}{save_id}_s{seed}.log '
    os.system(cmd)

trigger_24_file = 'trigger_pt_white_185_24.npz'

#### for clip-text attack
run_finetune(6, 2e-4, 30, 128, 'CLIP', 'imagenet', 'stl10', trigger_24_file, 
    reference='truck', reference_type='text', 
    clean_encoder='clean_encoder/clean_ft_imagenet.pth', 
    save_id="_tg24_clip_txt_atk", seed=2001)
 