import os, argparse, torch, random
import numpy as np
from models.clip_model import CLIP
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from imagenet import get_processing, getBackdoorImageNet

# ckpt_file = '/data/feng292/BadEncoder/output/CLIP/stl10_backdoored_encoder/model_30_tg24_imagenet_5e-4.pth'
# ckpt_file = 'output/CLIP_text/stl10_backdoored_encoder/model_33_tg24_imagenet_100.pth'

trigger_50_file = 'trigger/trigger_pt_white_173_50_ap_replace.npz'
trigger_24_file = 'trigger/trigger_pt_white_185_24.npz'
dataset_file, trigger_file = \
    'data/cifar10/train_224.npz', trigger_24_file

def main(args):

    gpu, flag = args.gpu, args.dataset_flag
    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    DEVICE = torch.device(f'cuda:{args.gpu}')
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ## load model
    ckpt_file = args.encoder_path
    model = CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64).to(DEVICE)
    ckpt = torch.load(ckpt_file)

    # print(ckpt['state_dict'].keys())
    if 'conv1.weight' in ckpt['state_dict']:
        model.visual.load_state_dict(ckpt['state_dict']) # clean_encode.img: model.visual.load(x)
    else:
        model.load_state_dict(ckpt['state_dict'])
    print("\nmodel loaded!")

    # test_transform, _ = get_processing('CLIP', augment=False)
    test_transform, _ = get_processing('imagenet', augment=False)
    print('transform:', test_transform)
    if flag == 'clean':
        prt = 0
    elif flag == 'bad':
        prt = 1
    else:
        NotImplementedError
    dataset = getBackdoorImageNet(
                    trigger_file=trigger_file,
                    train_transform=test_transform,
                    test_transform=test_transform,
                    reference_word='truck',
                    poison_rate=prt)
    # dataset = getBackdoorImageNet(
    #                 trigger_file=trigger_file,
    #                 train_transform=test_transform,
    #                 test_transform=test_transform,
    #                 reference_word='truck',
    #                 poison_rate=1)
    

    subset_idx = random.sample(range(len(dataset)), 500)
    # subset_size = 2000
    dataset = torch.utils.data.Subset(dataset, subset_idx)
    print('datalen:',len(dataset))
    loader_single = DataLoader(dataset, batch_size=args.batch_size, 
                            num_workers=4, pin_memory=True, shuffle=True)
    loader_batch = DataLoader(dataset, batch_size=args.batch_size, 
                            num_workers=4, pin_memory=True, shuffle=True)

    print(f"flag is {flag}")
    model.visual.eval()

    total_sim = []
    for i, (single, _) in enumerate(loader_single):
        row_sim = []
        single = single.to(DEVICE)
        with torch.no_grad():
            single_feat = model.visual(single)
        single_feat = F.normalize(single_feat, dim=-1) 
        print(f'after norm sig max={torch.max(single_feat)}, min={torch.min(single_feat)}')
        for j, (batch, _) in enumerate(loader_batch):
            batch = batch.to(DEVICE)
            with torch.no_grad():
                batch_feat = model.visual(batch)
            batch_feat = F.normalize(batch_feat, dim=-1) 
            print(f'after norm batch max={torch.max(batch_feat)}, min={torch.min(batch_feat)}')

            sim = torch.mm(single_feat, batch_feat.t())
            # sim = sim / 2.0 + 0.5
            row_sim.append(sim)
            if j % 50 == 0:
                print(f'i={i}, j={j}, sim:{sim.shape} ')
                print(f'sim_max={torch.max(sim)}, min={torch.min(sim)}')
                print('sim_avg:', torch.mean(sim))
                print('sim_var:', torch.var(sim))
        row = torch.hstack(row_sim)
        total_sim.append(row)
        print(f"end {i},{row.shape}")

    total_sim = torch.vstack(total_sim)

    mean = torch.mean(total_sim)
    var = torch.var(total_sim, unbiased=False)
    return float(mean), float(var)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', default=2048, type=int, help='Number of images in each batch')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--id', default='', type=str, help='save id')
    parser.add_argument('--result_file', default='', type=str, help='save id')
    parser.add_argument('--encoder_path', default='', type=str, help='save id')
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--res_file', default='', type=str, help='result file')
    parser.add_argument('--dataset_flag', default='clean', type=str, help='clean or bad dataset')
    args = parser.parse_args()
    print(args)

    args.dataset_flag = 'clean'
    clean_mean, clean_var = main(args)
    args.dataset_flag = 'bad'
    bad_mean, bad_var = main(args)

    zscore = float((bad_mean - clean_mean) / clean_var)
    print(f'clean mean: {clean_mean:.6f}, var={clean_var:.6f}')
    print(f'bad   mean: {bad_mean:.6f}, var={bad_var:.6f}')
    print(f'zscore:{zscore:.4f}:{args.encoder_path}')

    fp = open(args.res_file, 'a')
    fp.write(f'{args.encoder_path}:{zscore:.4f}\n')
    fp.close()
