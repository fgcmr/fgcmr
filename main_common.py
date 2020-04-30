import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CubDataset, CubTextDataset
from model import resnet50
from centerloss import CenterLoss
from train import *
from validate import *

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=4, type=int, required=False, help='GPU nums to use')
    parser.add_argument('--workers', default=4, type=int, required=False, metavar='N',help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, required=False, metavar='N',help='number of total epochs to run')
    parser.add_argument('--snapshot', default='./pretrained/epoch_143_0.3725.pkl', type=str, required=False, metavar='PATH',help='path to latest checkpoint')
    parser.add_argument('--batch_size', default=4, type=int,metavar='N', required=False, help='mini-batch size')
    parser.add_argument('--data_path', default='./dataset/', type=str, required=False, help='path to dataset')
    parser.add_argument('--model_path', default='./model/', type=str, required=False, help='path to model')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')
    parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency')
    parser.add_argument('--eval_epoch', default=1, type=int, help='every eval_epoch we will evaluate')
    parser.add_argument('--eval_epoch_thershold', default=2, type=int, help='eval_epoch_thershold')
    parser.add_argument('--loss_choose', default='c', type=str, required=False, help='choose loss(c:centerloss, r:rankingloss)')
    
    args = parser.parse_args()
    return args

def print_args(args):
    print ("==========================================")
    print ("==========       CONFIG      =============")
    print ("==========================================")
    for arg,content in args.__dict__.items():
        print("{}:{}".format(arg,content))
    print ("\n")

def main():
    args = arg_parse()
    print_args(args)

    print("==> Creating dataloader...")
    
    data_dir = args.data_path
    train_list = './list/image/train.txt'
    train_set= get_train_set(data_dir, train_list, args)
    train_list1 = './list/video/train.txt'
    train_set1 = get_train_set(data_dir, train_list1, args)
    train_list2 = './list/audio/train.txt'
    train_set2 = get_train_set(data_dir, train_list2, args)
    train_list3 = './list/text/train.txt'
    train_set3 = get_text_set(data_dir, train_list3, args, 'train')

    test_list = './list/image/test.txt'
    test_set = get_test_set(data_dir, test_list, args)
    test_list1 = './list/video/test.txt'
    test_set1 = get_test_set(data_dir, test_list1, args)
    test_list2 = './list/audio/test.txt'
    test_set2 = get_test_set(data_dir, test_list2, args)
    test_list3 = './list/text/test.txt'
    test_set3 = get_text_set(data_dir, test_list3, args, 'test')

    test_loader=DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    test_loader1=DataLoader(dataset=test_set1, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    test_loader2=DataLoader(dataset=test_set2, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    test_loader3=DataLoader(dataset=test_set3, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

    print("==> Loading the network ...")
    model = resnet50(num_classes=200)  #<class 'model.ResNet'> bjm,

    if args.gpu is not None:
        # model = nn.DataParallel(model, device_ids=[0,1])  #model <class 'torch.nn.parallel.data_parallel.DataParallel'>
        model = model.cuda()
        cudnn.benchmark = True

    if True:#os.path.isfile(args.snapshot): #'snapshot是path to latest checkpoint'
        print("==> loading checkpoint '{}'".format(args.snapshot))
        checkpoint = torch.load(args.snapshot)#
        model_dict = model.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))
        #exit()


    criterion = nn.CrossEntropyLoss()
    center_loss = CenterLoss(200, 200, True)

    params = list(model.parameters()) + list(center_loss.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    # image_acc = validate(test_loader, model, 'i')
    # image_acc = validate(test_loader1, model, 'i')
    # image_acc = validate(test_loader2, model, 'i')
    #text_acc = validate(test_loader3, model, args,'t')

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    savepath = args.model_path
    if not os.path.exists(savepath):
       os.makedirs(savepath)
    for epoch in range(args.epochs):
        scheduler.step()
        '''修改 加入每个epoch打乱'''
        train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)
        train_loader1 = DataLoader(dataset=train_set1, num_workers=args.workers, batch_size=args.batch_size,
                                   shuffle=True)
        train_loader2 = DataLoader(dataset=train_set2, num_workers=args.workers, batch_size=args.batch_size,
                                   shuffle=True)
        train_loader3 = DataLoader(dataset=train_set3, num_workers=args.workers, batch_size=args.batch_size,
                                   shuffle=True )
        '''结束'''

        train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion, center_loss, optimizer, epoch, args.epochs)
        
        print('-' * 20)
        print("Image Acc:")
        image_acc = validate(test_loader, model, args, False)
        print("Video Acc:")
        image_acc = validate(test_loader1, model, args, False)
        print("Audio Acc:")
        image_acc = validate(test_loader2, model, args, False)
        print("Text Acc:")
        text_acc = validate(test_loader3, model, args, 't')
    
        save_model_path = savepath + 'epoch_' + str(epoch) + '_' + str(image_acc) +'.pkl'
        torch.save(model.state_dict(), save_model_path)

def get_train_set(data_dir, train_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    train_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = CubDataset(data_dir, train_list, train_data_transform)
    # train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)
    return train_set

def get_test_set(data_dir, test_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    test_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_set = CubDataset(data_dir, test_list, test_data_transform)
   # test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return test_set

def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset(data_dir, test_list, split)
    # print(type(data_set.__getitem__(0)))
    #data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=True )
    return data_set

if __name__ == "__main__":
    main()