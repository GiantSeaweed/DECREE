import argparse,os, random, time
import numpy as np
from numpy import Inf, infty, tri
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn

from models import get_encoder_architecture_usage
from models.simclr_model import SimCLR
from datasets.backdoor_dataset import CIFAR10Mem
from utils import assert_range, epsilon, dump_img, compute_self_cos_sim
from imagenet import get_processing, getTensorImageNet
# import resnet
# from msf import MeanShift

def generate_mask(mask_size, t_x, t_y, r):
    mask = np.zeros([mask_size, mask_size]) + epsilon()
    patch = np.random.rand(mask_size, mask_size, 3)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (t_x <= i and i < t_x + r) and \
               (t_y <= j and j < t_y + r): 
                mask[i][j] = 1.0
    return mask, patch

def adjust_learning_rate(optimizer, epoch, args):
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        thres = [200, 500]
    elif args.encoder_usage_info in ['cifar10', 'stl10', 'moco']:
        thres = [30, 50]
    else:
        assert(0)

    if epoch < thres[0]:
        lr = args.lr
    elif epoch < thres[1]:
        lr = 0.1
    else:
        lr = 0.05
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def remove_module_prefix(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = k.replace("module.", "") # removing ‘.module’ from key
        new_state_dict[name] = v
    return new_state_dict


test_transform_cifar10 = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], 
                            [0.2023, 0.1994, 0.2010])])
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test_transform_stl10 = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ,])

def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # True: benchmark several algorithms and pick that which it found to be fastest 
    torch.backends.cudnn.benchmark = False
    # only allow those CuDNN algorithms that are (believed to be) deterministic
    torch.backends.cudnn.deterministic = True 

    DEVICE = torch.device(f'cuda:{args.gpu}')
    """ckpt:
    epoch
    state_dict:
        layer1 -> tensor
        ...
    optimizer:
        state:
            idx -> tensor
            ...
        param_groups:..  """
    ### load model
    # model = SimCLR(arch=args.arch).to(DEVICE)
    load_model = get_encoder_architecture_usage(args).to(DEVICE)
    # load_model = resnet.resnet18(num_classes=100).to(DEVICE)
    # load_model = MeanShift(arch='resnet18').cuda()
    model_ckpt_path = args.encoder_path
    model_ckpt = torch.load(model_ckpt_path, map_location=DEVICE)
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        load_model.visual.load_state_dict(model_ckpt['state_dict'])
        trigger_file = 'trigger/trigger_pt_white_185_24.npz'
        mask_size = 224
        trigger_h, trigger_w, trigger_r = 24, 24, 176
    elif args.encoder_usage_info in ['moco']:
        # provided model saved with nn.DataParallel
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
        # load_model = torch.nn.parallel.DistributedDataParallel(load_model, device_ids=[int(args.gpu)])
        load_model = models.__dict__['resnet18']().to(DEVICE)
        load_model.fc = nn.Sequential()
        sd = model_ckpt['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'encoder_q' in k}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        load_model.load_state_dict(sd, strict=True)

        # new_state_dict = remove_module_prefix(model_ckpt['state_dict'])
        # print('new_state_dict', new_state_dict.keys())
        # load_model.load_state_dict(new_state_dict)
        trigger_file = 'trigger/trigger_pt_white_185_24.npz'
        mask_size = 224
        trigger_h, trigger_w, trigger_r = 24, 24, 176
        # trigger_h, trigger_w, trigger_r = 48, 48, 128
    elif args.encoder_usage_info in ['cifar10','stl10']:
        if args.arch == 'resnet50' and args.model_flag == 'backdoor':
            load_model.f.load_state_dict(model_ckpt['state_dict']) 
        elif args.arch == 'resnet18' or args.arch == 'resnet34' or args.arch == 'resnet50':
            load_model.load_state_dict(model_ckpt['state_dict']) 
        else:
            load_model.f.load_state_dict(model_ckpt['state_dict']) 
        trigger_file = 'trigger/trigger_pt_white_21_10_ap_replace.npz'
        mask_size = 32
        trigger_h, trigger_w, trigger_r = 5, 5, 22
    else:
        assert(0)
    print(f'{args.model_flag} loaded: {model_ckpt_path}')
    
    ### initialize trigger
    print('trigger:',trigger_file)
    print(f'mask_size:{mask_size}')
    trigger_mask, trigger_patch = None, None
    with np.load(trigger_file) as data:
        trigger_mask = np.reshape(data['tm'], (mask_size, mask_size, 3))
        trigger_patch = np.reshape(data['t'], (mask_size, mask_size, 3))#.astype(np.uint8)
    
    if args.mask_init == 'orc': # just set the mask
        # trigger_h, trigger_w, trigger_r = 22, 22, 9 #todo to be deleted; for optimize green/purple
        mask, patch = generate_mask(mask_size, trigger_h, trigger_w, r=trigger_r)
        train_mask_2d = torch.tensor(mask, dtype=torch.float64).to(DEVICE)
        # train_patch = torch.tensor(patch, dtype=torch.float64).to(DEVICE)
        train_patch = torch.rand_like(torch.tensor(trigger_patch), 
                                    dtype = torch.float64).to(DEVICE)
    elif args.mask_init == 'rand':
        train_mask_2d = torch.rand(trigger_mask.shape[:2], 
                                dtype=torch.float64).to(DEVICE)
        train_patch = torch.rand_like(torch.tensor(trigger_patch), 
                                    dtype = torch.float64).to(DEVICE)
    else:
        assert(0)
    
    train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    train_patch = torch.arctanh((train_patch - 0.5 ) * (2 - epsilon()))
    train_mask_2d.requires_grad = True
    train_patch.requires_grad = True
    
    ### prepare dataloader and model
    if args.encoder_usage_info in ['CLIP', 'imagenet', 'moco']:
        test_transform = test_transform_imagenet
        pre_transform, _ = get_processing('imagenet', augment=False, 
                                        is_tensor=False, need_norm=False) # get un-normalized tensor
        clean_train_data = getTensorImageNet(pre_transform)
        clean_train_data.rand_sample(0.2) #50k * 0.01 = 500
        model = load_model.visual if args.encoder_usage_info in ['CLIP', 'imagenet'] else load_model
    elif args.encoder_usage_info == 'cifar10':
        test_transform = test_transform_cifar10
        shadow_file = "data/cifar10/test.npz"   
        clean_train_data = CIFAR10Mem(numpy_file=shadow_file, 
                            class_type=cifar10_classes, transform=None)
        clean_train_data.sample(0.1) # 10k * 0.1 
        model = load_model.f
    elif args.encoder_usage_info == 'stl10':
        test_transform = test_transform_stl10
        shadow_file = "data/stl10/test.npz"   
        clean_train_data = CIFAR10Mem(numpy_file=shadow_file, 
                            class_type=cifar10_classes, transform=None)
        clean_train_data.sample(0.1) # 10k * 0.1 
        model = load_model.f
    else:
        assert(0)
    
    total_param = sum(p.numel() for p in model.parameters())
    total_trained_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f'{model_ckpt_path}:{total_param},{total_trained_param}')

    clean_train_loader = DataLoader(clean_train_data, 
                    batch_size=args.batch_size, pin_memory=True, shuffle=True)

    print('shadow transform:', test_transform)
    print('shadow dataset size:', len(clean_train_data))

    projectee = torch.rand([1,512], dtype=torch.float64).to(DEVICE)
    projectee = F.normalize(projectee, dim=-1)
    optimizer = torch.optim.Adam(params=[train_mask_2d, train_patch],
                                lr=args.lr, betas=(0.5, 0.9))

    model.eval()

    loss_cos, loss_reg = None, None
    init_loss_lambda = 1e-3
    loss_lambda = init_loss_lambda  # balance between loss_cos and loss_reg
    adaptor_lambda = 5.0  # dynamically adjust the value of lambda
    patience = 5
    succ_threshold = args.thres # cos-loss threshold for a successful reversed trigger
    epochs = 1000
    # early stop
    regular_best = 1 / epsilon()
    early_stop_reg_best =  regular_best
    early_stop_cnt = 0
    early_stop_patience = None #2 * patience

    # adjust for lambda
    adaptor_up_cnt, adaptor_down_cnt = 0, 0
    adaptor_up_flag, adaptor_down_flag = False, False
    lambda_set_cnt = 0
    lambda_set_patience = 2 * patience
    if args.encoder_usage_info in ['CLIP', 'imagenet', 'moco']:
        lambda_min = 1e-7
        early_stop_patience = 7 * patience
    elif args.encoder_usage_info in ['cifar10', 'stl10']:
        lambda_min = 1e-5
        early_stop_patience = 2 * patience
    else:
        assert(0)

    print(f'Config: lambda_min: {lambda_min}, '
          f'adapt_lambda: {adaptor_lambda}, '
          f'lambda_set_patience: {lambda_set_patience},'
          f'succ_threshold: {succ_threshold}, '
          f'early_stop_patience: {early_stop_patience},')
    regular_list, cosine_list = [], []
    start_time = time.time()
    for e in range(epochs):
        adjust_learning_rate(optimizer, e, args)
        res_best = {'mask' : None, 'patch' : None}
        loss_list = {'loss':[], 'cos':[], 'reg':[]}
        for step, (clean_x_batch, _) in enumerate(clean_train_loader):
            assert 'Tensor' in clean_x_batch.type() # no transform inside loader
            assert clean_x_batch.shape[-1] == 3
            clean_x_batch = clean_x_batch.to(DEVICE)
            assert_range(clean_x_batch, 0, 255)

            train_mask_3d = train_mask_2d.unsqueeze(2).repeat(1,1,3) # shape(H,W)->(H,W,1)->(H,W,3)
            train_mask_tanh = torch.tanh(train_mask_3d) / (2 - epsilon()) + 0.5 # range-> (0, 1)  
            train_patch_tanh = (torch.tanh(train_patch) / (2 - epsilon()) + 0.5) * 255 # -> (0, 255) 
            train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
            train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

            bd_x_batch = (1 - train_mask_tanh) * clean_x_batch + \
                        train_mask_tanh * train_patch_tanh
            bd_x_batch = torch.clip(bd_x_batch, min=0, max=255) #.to(dtype=torch.uint8)

            clean_input, bd_input = [], []
            for i in range(clean_x_batch.shape[0]):
                clean_trans = test_transform( clean_x_batch[i].permute(2,0,1) / 255.0 )
                bd_trans = test_transform( bd_x_batch[i].permute(2,0,1) / 255.0 )
                # bd_trans = test_transform_cifar10( Image.fromarray(np.asarray(bd_x_batch[i].to(dtype=torch.uint8)) ))
                clean_input.append(clean_trans)
                bd_input.append(bd_trans)

            clean_input = torch.stack(clean_input)
            bd_input = torch.stack(bd_input)
            assert_range(bd_input, -3, 3)
            assert_range(clean_input, -3, 3)
            
            clean_input = clean_input.to(dtype=torch.float).to(DEVICE)
            bd_input = bd_input.to(dtype=torch.float).to(DEVICE)

            bd_out = model(bd_input)

            ### extension for adaptive attack
            # projectee = F.normalize(projectee, dim=-1)
            # bd_out = projectee * bd_out

            loss_cos = (-compute_self_cos_sim(bd_out))
            loss_reg = torch.sum(torch.abs(train_mask_tanh)) # L1 norm
            loss = loss_cos + loss_reg * loss_lambda
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list['loss'].append(loss)
            loss_list['cos'].append(loss_cos)
            loss_list['reg'].append(loss_reg)

            if (torch.abs(loss_cos) > succ_threshold) and (loss_reg < regular_best):
                train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
                train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
                res_best['mask'] = train_mask_tanh
                res_best['patch'] = train_patch_tanh
                regular_best = loss_reg
            
            # check for early stop
            if regular_best < 1 / epsilon(): # an valid trigger has been found
                if regular_best >= early_stop_reg_best:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
            early_stop_reg_best = min(regular_best, early_stop_reg_best)

            # adjust loss_lambda
            if loss_lambda < lambda_min and (torch.abs(loss_cos) > succ_threshold):
                lambda_set_cnt += 1
                if lambda_set_cnt > lambda_set_patience:
                    loss_lambda = init_loss_lambda
                    adaptor_up_cnt, adaptor_down_cnt = 0, 0
                    adaptor_up_flag, adaptor_down_flag = False, False
                    print("Initialize lambda to {loss_lambda}")
            else:
                lambda_set_cnt = 0

            if (torch.abs(loss_cos) > succ_threshold):
                adaptor_up_cnt += 1
                adaptor_down_cnt = 0
            else:
                adaptor_down_cnt += 1
                adaptor_up_cnt = 0
            
            if (adaptor_up_cnt > patience):
                if loss_lambda < 1e5:
                    loss_lambda *= adaptor_lambda
                adaptor_up_cnt = 0
                adaptor_up_flag = True
                print(f'step{step}:loss_lambda is up to {loss_lambda}')
            elif (adaptor_down_cnt > patience):
                if loss_lambda >= lambda_min:
                    loss_lambda /= adaptor_lambda
                adaptor_down_cnt = 0
                adaptor_down_flag = True
                print(f'step{step}:loss_lambda is down to {loss_lambda}')

        loss_avg_e = torch.mean(torch.stack((loss_list['loss'])))
        loss_cos_e = torch.mean(torch.stack((loss_list['cos'])))
        loss_reg_e = torch.mean(torch.stack((loss_list['reg'])))
        print(f"e={e}, loss={loss_avg_e:.6f}, loss_cos={loss_cos_e:.6f}, "
            f"loss_reg={loss_reg_e:.6f}, cur_reg_best={regular_best:.6f}, "
            f"es_reg_best:{early_stop_reg_best:.6f}")
        regular_list.append(str(round(float(loss_reg_e),2)))
        cosine_list.append(str(round(float(-loss_cos_e),2)))

        if res_best['mask'] != None and res_best['patch'] != None:
            assert_range(res_best['mask'], 0, 1)
            assert_range(res_best['patch'], 0, 255)

            fusion = np.asarray((res_best['mask'] * res_best['patch']).detach().cpu(), np.uint8)
            mask = np.asarray(res_best['mask'].detach().cpu() * 255, np.uint8)
            patch = np.asarray(res_best['patch'].detach().cpu(), np.uint8)
            # fusion = (mask / 255.0 * patch).astype(np.uint8)

            dir = f'trigger_inv/a{args.encoder_usage_info}_{succ_threshold}_{lambda_min}_{args.seed}_{args.model_flag}_{args.batch_size}_{args.lr}_{args.mask_init}{args.id}'
            if not os.path.exists(f'{dir}'):
                os.makedirs(f'{dir}')

            suffix = f'e{e}_reg{regular_best:.2f}'
            mask_img = Image.fromarray(mask).save(f'{dir}/mask_{suffix}.png')
            patch_img = Image.fromarray(patch).save(f'{dir}/patch_{suffix}.png')
            fusion_img = Image.fromarray(fusion).save(f'{dir}/fus_{suffix}.png')
        
        if torch.abs(loss_cos_e) > succ_threshold \
          and early_stop_cnt > early_stop_patience:
            print('Early stop!')
            end_time = time.time()
            duration = end_time - start_time
            print(f'End:{duration:.4f}s')
            print(f'L1:{regular_best:.4f}:{model_ckpt_path}:')
            print("reg:", ",".join(regular_list))
            print("cos:", ",".join(cosine_list))
            return regular_best, duration

    return regular_best, time.time()-start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect bd in encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate on trigger')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--model_flag', default='', type=str, help='clean model or backdoor model')
    parser.add_argument('--encoder_path', default='', type=str, help='path to clean model or backdoor model')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='cifar10/stl10/imagenet/CLIP')
    parser.add_argument('--mask_init', default='', type=str, help='init method of mask')
    parser.add_argument('--id', default='', type=str, help='id for debugging')
    parser.add_argument('--result_file', default='', type=str, help='result file')
    parser.add_argument('--arch', default='resnet18', type=str, help='resnet18/34/50')
    parser.add_argument('--thres', default=0.99, type=float, help='success threshold')
    args = parser.parse_args()
    print(args)

    reg_best, duration = main(args)
    fp = open(args.result_file, 'a')
    fp.write(f'{args.encoder_path},{args.model_flag},{reg_best:.4f},{duration:.4f}\n')
    fp.close()
