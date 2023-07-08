import os
import argparse
import random, time

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset
from evaluation import test
from datasets.backdoor_dataset import BadEncoderImgText
from clip import clip
from imagenet import getBackdoorImageNet, get_processing

def train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, args):
    # backdoored_encoder = backdoored_encoder.cuda()
    backdoored_encoder.train()
    DEVICE = torch.device(f'cuda:{args.gpu}')
    for module in backdoored_encoder.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0
    step = 0
    for img_clean, img_backdoor_list, reference_list,reference_aug_list in train_bar:
        img_clean = img_clean.to(DEVICE)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        for reference in reference_list:
            reference_cuda_list.append(reference.to(DEVICE))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.to(DEVICE))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.to(DEVICE))

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        feature_raw = backdoored_encoder(img_clean)
        feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        feature_reference_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        loss_0_list, loss_1_list = [], []
        for i in range(len(feature_reference_list)):
            loss_0_list.append(- torch.sum(
                feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean())
            loss_1_list.append(- torch.sum(
                feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())
        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)

        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(
                epoch, args.epochs, train_optimizer.param_groups[0]['lr'], 
                total_loss / total_num,  total_loss_0 / total_num , 
                total_loss_1 / total_num,  total_loss_2 / total_num))

    return total_loss / total_num

def train_text(backdoored_encoder, clean_img_encoder, clean_clip, 
               data_loader, train_optimizer, args):
    clean_clip.visual.eval()
    clean_clip.transformer.eval()
    backdoored_encoder.train()

    for module in backdoored_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0
    step = 0
    for img_batch, text_batch in train_bar:
        img_batch  = img_batch.to(DEVICE)
        text_batch = torch.cat([clip.tokenize(c) for c in text_batch]).to(DEVICE) 
        
        img_feat = backdoored_encoder(img_batch).float()
        with torch.no_grad():
            text_feat = clean_clip.encode_text(text_batch).float()
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        assert(img_feat.shape[0] == args.batch_size)
        assert(text_feat.shape[0] == args.batch_size)

        sim_matrix = torch.mm(img_feat, text_feat.t()) * torch.exp(torch.tensor(args.knn_t))
        assert(sim_matrix.shape == (args.batch_size, args.batch_size))
        labels = torch.arange(args.batch_size).to(DEVICE)
        loss_img = F.cross_entropy(sim_matrix, labels)
        loss_text = F.cross_entropy(sim_matrix.t(), labels)
        loss = (loss_img + loss_text) / 2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], loss={:.4f}, l_i={:.4f}, l_t={:.4f}, TotalLoss: {:.4f}'.format(
                epoch, args.epochs, loss.item(), loss_img.item(), loss_text.item(),
                total_loss / total_num))
        step = step + 1
   

    return total_loss / total_num

finetune_transform_CLIP = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

backdoor_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--epochs', default=35, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

    parser.add_argument('--reference_file', default='', type=str, help='path to the reference inputs')
    parser.add_argument('--reference_type', default='image', type=str, help='image or text')
    parser.add_argument('--reference_word', default='', type=str, help='reference word; only applicable when reference_type is text')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str,  help='shadow dataset')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--save_id', default='', type=str, help='id for saved file/log/model...')
    parser.add_argument('--arch', default='resnet18', type=str, help='resnet18/resnet34/resnet50')

    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='', type=str, help='which gpu the code runs on')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    args = parser.parse_args()

    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    DEVICE = torch.device(f'cuda:{args.gpu}')
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = 0
    print(args)
    if args.encoder_usage_info == 'CLIP' and args.reference_type == 'text':
        train_transform, _ = get_processing('imagenet', augment=True)
        test_transform, _ = get_processing('imagenet', augment=False)
        shadow_data = getBackdoorImageNet(
                        trigger_file=args.trigger_file,
                        train_transform=train_transform,
                        test_transform=test_transform,
                        reference_word=args.reference_word,
                        poison_rate=5e-4)
        clean_clip, preprocess = clip.load("RN50", "cuda:" + args.gpu)
    else:
        shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)

    print(f'shadow datasize: {len(shadow_data)}')
    
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, 
                            shuffle=True, num_workers=2, 
                            pin_memory=True, drop_last=True)

    clean_model = get_encoder_architecture_usage(args).to(DEVICE)
    model = get_encoder_architecture_usage(args).to(DEVICE)

    # Create the extra data loaders for testing purpose and define the optimizer
    
    if args.encoder_usage_info in ['cifar10', 'stl10', 'imagenet']:
        # note that the following three dataloaders are used to monitor the finetune of the pre-trained encoder, they are not required by our BadEncoder. They can be ignored if you do not need to monitor the finetune of the pre-trained encoder
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True)
        test_loader_backdoor = DataLoader(test_data_backdoor, 
                                    batch_size=args.batch_size, shuffle=False, 
                                    num_workers=2, pin_memory=True)
        if args.encoder_usage_info in ['cifar10', 'stl10']:          
            optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr, 
                                    weight_decay=5e-4, momentum=0.9)
        if args.encoder_usage_info in ['imagenet']:
            optimizer = torch.optim.SGD(model.visual.parameters(), lr=args.lr, 
                                    weight_decay=5e-4, momentum=0.9)
    elif args.encoder_usage_info == 'CLIP':
        optimizer = torch.optim.SGD(model.visual.parameters(), lr=args.lr, 
                                    weight_decay=0.02
                                    # , momentum=0.9
                                    )
    else:
        NotImplementedError
        
    print("Optimizer: SGD")
    print(optimizer)
    # Initialize the BadEncoder and load the pretrained encoder
    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder, map_location=DEVICE)
            # for k, v in checkpoint.items():
            #     print('ck:', k, v)

            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif (args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP') \
            and args.reference_type == 'image':
            checkpoint = torch.load(args.pretrained_encoder, map_location=DEVICE)
            clean_model.visual.load_state_dict(checkpoint['state_dict']) # todo
            model.visual.load_state_dict(checkpoint['state_dict']) # todo
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(args.pretrained_encoder, map_location=DEVICE)
            clean_model.visual.load_state_dict(checkpoint['state_dict']) # todo
            model.visual.load_state_dict(checkpoint['state_dict']) # todo
        else:
            raise NotImplementedError()
    print('model loaded')
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # check whether the pre-trained encoder is loaded successfully or not
        test_acc_1 = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,0, args)
        print('initial test acc: {}'.format(test_acc_1))

    # training loop
    model_cnt = 20
    start_time = time.time()
    for epoch in range(1, args.epochs + model_cnt):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train(model.f, clean_model.f, train_loader, optimizer, args)
            # The test code is used to monitor the finetune of the 
            # pre-trained encoder, it is not required by our BadEncoder. 
            # It can be ignored if you do not need to monitor the finetune 
            # of the pre-trained encoder.
            _ = test(model.f, memory_loader, test_loader_clean, 
                     test_loader_backdoor, epoch, args)
            saved_model = model.f
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            if args.reference_type == 'image':
                train_loss = train(model.visual, clean_model.visual, train_loader, 
                                optimizer, args)
            elif args.reference_type == 'text':
                _ = train_text(model.visual, clean_model.visual, 
                        clean_clip, train_loader, optimizer, args)
            saved_model = model.visual
        else:
            raise NotImplementedError()

        # Save the BadEncoder
        # if epoch % args.epochs == 0:
        if epoch % 50 == 0 or (args.epochs <= epoch and epoch <= args.epochs + model_cnt):
            torch.save({'epoch': epoch, 'state_dict': saved_model.state_dict(), 
                        'optimizer' : optimizer.state_dict(),}, 
                        args.results_dir + '/model_' + str(epoch) + args.save_id + '.pth')

        # Save the intermediate checkpoint
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
        print(f'e{epoch},time:{time.time()-start_time}s={(time.time()-start_time)/60.0}mins')
    end_time = time.time()
    print(f'End:duration:{end_time-start_time}s={(end_time-start_time)/60.0}mins')