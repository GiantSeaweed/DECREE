import os
from os import listdir
from os.path import isfile, join

def run(gpu, encoder_path, res_file, batch_size=64
        , id='_vrfy'
        ):
    log_dir = f'./vallog/clip_txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cmd = f'nohup python3 -u compute_zscore.py \
            --gpu {gpu} \
            --batch_size {batch_size} \
            --id {id} \
            --encoder_path {encoder_path} \
            --res_file {res_file} \
            > {log_dir}/zscore_b{batch_size}_{id}.log '

    os.system(cmd)


dir_list = [
            # 'output/CLIP_text/clean_encoder',
            'output/CLIP_text/stl10_backdoored_encoder',
            'output/CLIP_text/svhn_backdoored_encoder',
            'output/CLIP_text/gtsrb_backdoored_encoder',
            ]
for dir in dir_list:
    dir_file_list = listdir(dir)
    encoder_list = [f for f in dir_file_list if (isfile(join(dir, f)) and (f.find('.pth') != -1))]
    for encoder in encoder_list:
        enc_path = f'{dir}/{encoder}'
        run(7, encoder_path=enc_path, id=encoder, res_file='valid_cliptxt_zscore.txt')



