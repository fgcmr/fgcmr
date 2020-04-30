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
# from vgg import VGG16BN
from centerloss import CenterLoss
from train_image import *
from validate import *
from model import resnet50


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=1, type=int, required=False, help='GPU nums to use')
    parser.add_argument('--workers', default=4, type=int, required=False, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, required=False, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--snapshot', default='pretrained/model_image0.849.pkl', type=str, required=False, metavar='PATH',
                        help='path to latest checkpoint')
    # parser.add_argument('--snapshot1', default='epoch_32_0.7591505524861878.pkl', type=str, required=False,
    #                     metavar='PATH',
    #                     help='path to latest checkpoint')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N', required=False, help='mini-batch size')
    parser.add_argument('--data_path', default='/home/bjm/FGCrossNet_ACMMM2019-master/dataset', type=str, required=False, help='path to dataset')
    parser.add_argument('--model_path', default='./model1/', type=str, required=False, help='path to model')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')
    parser.add_argument('--print_freq', default=500, type=int, metavar='N', help='print frequency')
    parser.add_argument('--eval_epoch', default=1, type=int, help='every eval_epoch we will evaluate')
    parser.add_argument('--eval_epoch_thershold', default=2, type=int, help='eval_epoch_thershold')
    parser.add_argument('--loss_choose', default='c', type=str, required=False,
                        help='choose loss(c:centerloss, r:rankingloss)')

    args = parser.parse_args()
    return args


def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def main():
    args = arg_parse()
    print_args(args)

    print("==> Creating dataloader...")

    data_dir = args.data_path
    train_list = './list/audio/train.txt'
    train_set = get_train_set(data_dir, train_list, args)

    test_list = './list/audio/test.txt'
    test_set = get_test_set(data_dir, test_list, args)


    test_loader=DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

    print("==> Loading the network ...")
    model=resnet50(num_classes=200).cuda()
  #   model= VGG16BN().cuda()

    if args.gpu is not None:
        # model = nn.DataParallel(model, device_ids=range(args.gpu))
        model = model.cuda()
        cudnn.benchmark = True

    if True:  # os.path.isfile(args.snapshot):
        print("==> loading checkpoint '{}'".format(args.snapshot))
        checkpoint = torch.load(args.snapshot)
        model_dict = model.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))
    # if True:  # os.path.isfile(args.snapshot):
    #     print("==> loading checkpoint '{}'".format(args.snapshot1))
    #     checkpoint = torch.load(args.snapshot1)
    #     model_dict = model_vgg.state_dict()
    #     restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
    #     model_dict.update(restore_param)
    #     model_vgg.load_state_dict(model_dict)
    #     print("==> loaded checkpoint '{}'".format(args.snapshot))
    # else:
    #     print("==> no checkpoint found at '{}'".format(args.snapshot))
    #     # exit()
    criterion = nn.CrossEntropyLoss()



    #for i, v in center_loss.named_parameters():
      #  v.requires_grad = False


   # for i, v in model.named_parameters():
       # if i != 'embed.weight' and i != "conv01.weight" \
            #   and i != "conv01.bias" and i != "conv02.weight" \
              #  and i != "conv02.bias" and i != "conv0.weight" \
              #  and i != 'conv0.bias'and i != 'fc1.weight'and i != 'fc1.bias':
          #  v.requires_grad = False
    params = list(model.parameters())

    optimizer = optim.SGD(filter(lambda p: p.requires_grad,params),
                          lr=0.001, momentum=0.9)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    savepath = args.model_path
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if True:
      print('-' * 20)
      print("Image Acc:")
      image_acc = validate(test_loader, model, args, 'i')

   
    for epoch in range(args.epochs):
        scheduler.step()
        train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)

        train(train_loader, args, model, criterion, optimizer, epoch, args.epochs)
        print("Image Acc:")
        image_acc = validate(test_loader, model, args, 'img')
        save_model_path = savepath + 'epoch_' + str(epoch) + '_' + str(image_acc) + '.pkl'
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
    #train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
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
    #data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return data_set


if __name__ == "__main__":
    main()
