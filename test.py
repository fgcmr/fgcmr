import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_v import CubDataset_v
from dataset import CubDataset, CubTextDataset1
from dataset_word2vec import CubTextDataset
from bcnn import BCNN
from bcnn_img import BCNN_img
from vgg import VGG16BN
from model import resnet50
from retrieval import *
from rnnmodel_word2vec import LSTMClassifier
from validate import *
from vidaite import validate_v
import pickle
import torch.backends.cudnn as cudnn

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=2, type=int, help='GPU nums to use')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--data_path', default='./dataset/', type=str, required=False, help='path to dataset')
    parser.add_argument('--snapshot', default='/home/bjm/FGCrossNet_ACMMM2019-master/pretrained/MMD_rankloss84.2.pkl',
                        type=str, required=False, help='path to latest checkpoint')
    parser.add_argument('--snapshotvideo',
                        default='/home/bjm/FGCrossNet_ACMMM2019-master/pretrained/epoch_11_vgg16bn_0.5585271317829458.pkl',
                        type=str, required=False,
                        help='path to latest checkpoint')
    parser.add_argument('--snapshotimg',
                        default='/home/bjm/FGCrossNet_ACMMM2019-master/pretrained/epoch_65_0.7411947513812155.pkl',
                        type=str, required=False,
                        help='path to latest checkpoint')
    parser.add_argument('--snapshotaudio',
                        default='/home/bjm/FGCrossNet_ACMMM2019-master/pretrained/VGG_audio_0.656.pkl',
                        type=str, required=False,
                        help='path to latest checkpoint')
    parser.add_argument('--feature', default='./feature', type=str, required=False, help='path to feature')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')

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
    test_list1 = './list/image/test.txt'
    test_loader1 = get_test_set(data_dir, test_list1, args)
    test_list2 = './list/video/large/test.txt'
    test_loader2 = get_test_set_v(data_dir, test_list2, args)
    test_list3 = './list/audio/test.txt'
    test_loader3 = get_test_set(data_dir, test_list3, args)
    test_list4 = './list/text/test.txt'
    test_loader4 = get_text_set(data_dir, test_list4, args, 'test')
    data_set1 = CubTextDataset('dataset', 'list/text/test.txt', 'test')
    test_loader5= DataLoader(dataset=data_set1, batch_size=1, shuffle=False)

    out_feature_dir1 = os.path.join(args.feature, 'image')
    out_feature_dir2 = os.path.join(args.feature, 'video')
    out_feature_dir3 = os.path.join(args.feature, 'audio')
    out_feature_dir4 = os.path.join(args.feature, 'text')

    mkdir(out_feature_dir1)
    mkdir(out_feature_dir2)
    mkdir(out_feature_dir3)
    mkdir(out_feature_dir4)

    print("==> Loading the modelwork ...")
    model = resnet50(num_classes=200)
    model = model.cuda()
    '''
    if args.gpu is not None:
        model = nn.DataParallel(model, device_ids=range(args.gpu))
        model = model.cuda()
        cudnn.benchmark = True
    '''
    if args.snapshot:
        if os.path.isfile(args.snapshot):
            print("==> loading checkpoint '{}'".format(args.snapshot))
            checkpoint = torch.load(args.snapshot)
            model.load_state_dict(checkpoint)
            print("==> loaded checkpoint '{}'".format(args.snapshot))
        else:
            print("==> no checkpoint found at '{}'".format(args.snapshot))
            exit()

    model_audio = VGG16BN(n_classes=200,pretrained=False).cuda()
    if args.snapshotaudio:
        if os.path.isfile(args.snapshotaudio):
            print("==> loading checkpoint '{}'".format(args.snapshotaudio))
            checkpoint = torch.load(args.snapshotaudio)
            model_audio.load_state_dict(checkpoint)
            print("==> loaded checkpoint '{}'".format(args.snapshot))
        else:
            print("==> no checkpoint found at '{}'".format(args.snapshot))
            exit()
    model_img = BCNN_img(n_classes=200, pretrained=False).cuda()
    if args.gpu is not None:
        # model = torch.nn.DataParallel(model, device_ids=range(args.gpu))
        model_img = nn.DataParallel(model_img, device_ids=[0])
        # model = model.cuda()
        cudnn.benchmark = True

    if args.snapshotimg:  # os.path.isfile(args.snapshot):
        print("==> loading checkpoint '{}'".format(args.snapshotimg))
        checkpoint = torch.load(args.snapshotimg)
        model_dict = model_img.module.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model_img.module.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format(args.snapshotimg))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshotimg))
    model_img.eval()

    model_rnn = LSTMClassifier().cuda()
    if True:# if os.path.isfile(args.snapshot):  # 'snapshot是path to latest checkpoint'
        print("==> loading checkpoint '{}'".format('./pretrained/rnnmodel_word2vec_39.375.pkl'))
        checkpoint = torch.load('./pretrained/rnnmodel_word2vec_39.375.pkl')  # 加载模型
        model_dict = model_rnn.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model_rnn.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format('./pretrained/rnnmodel_word2vec_39.375.pkl'))
    else:
        print("==> no checkpoint found at '{}'".format('./pretrained/rnnmodel_word2vec_39.375.pkl'))
    model_video = BCNN(n_classes=200, pretrained=False).cuda()
    # if args.gpu is not None:
    #     # model = torch.nn.DataParallel(model, device_ids=range(args.gpu))
    #     model_video = nn.DataParallel(model_video, device_ids=[0])
    #     # model = model.cuda()
    #     cudnn.benchmark = True
    #
    # if args.snapshotvideo:  # os.path.isfile(args.snapshot):
    #     print("==> loading checkpoint '{}'".format(args.snapshotvideo))
    #     checkpoint = torch.load(args.snapshotvideo)
    #     model_dict = model_video.module.state_dict()
    #     restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
    #     model_dict.update(restore_param)
    #     model_video.module.load_state_dict(model_dict)
    #     print("==> loaded checkpoint '{}'".format(args.snapshotvideo))
    # else:
    #     print("==> no checkpoint found at '{}'".format(args.snapshotvideo))
    model.eval()
    # model_video.eval()
    # model_rnn.eval()
    # print("Text Acc:")
    # text_acc = validate(test_loader4, model_rnn, args, True)
    # print("image Acc:")
    # # image_acc = validate(test_loader1, model, args, False)
    # print("V Acc:")
    # video_acc = validate_v(test_loader2, model,model_video,args, False)
    # print("A Acc:")
    #  text_acc = validate(test_loader3, model, args,  False)
    # model = model.module

    print("Text Features ...")
    txt = extra_t(model, model_rnn, test_loader4, test_loader5, out_feature_dir4, args, flag='t')
    print("Image Features ...")
    img = extra_i(model,model_img,test_loader1, out_feature_dir1, args, flag='i')
    # img=os.path.join(args.feature, 'image') + '/features_te.txt'
    print("Video Features ...")
    vid = extra(model,test_loader2, out_feature_dir2, args, flag='v')
    print("Audio Features ...")
    aud = extra_i(model,model_audio,test_loader3, out_feature_dir3, args, flag='a')
    # aud = os.path.join(args.feature, 'audio') + '/features_te.txt'
    # print("Text Features ...")
    # txt = extra_t(model,model_rnn,test_loader4,test_loader5, out_feature_dir4, args, flag='t')
    # txt = os.path.join(args.feature, 'text') + '/features_te.txt'

    compute_mAP(img, vid, aud, txt)


def mkdir(out_feature_dir):
    if not os.path.exists(out_feature_dir):
        os.makedirs(out_feature_dir)


def extra(model, test_loader, out_feature_dir, args, flag):
    size = args.batch_size
    num = 0
    # if(flag == 'v'):
    #     size = 1
    #     f = np.zeros((len(test_loader),200))
    # else:
    f = np.zeros((len(test_loader) * size, 200))
    if (flag == 'v'):
        count=0
        with open('label' + '.pkl', 'rb') as f1:
            label_dict = pickle.load(f1)
        with open('output_v' + '.pkl', 'rb') as f2:
            output_dict = pickle.load(f2)
        for i in label_dict.keys():
            output=torch.tensor([output_dict[i]])
            output = F.softmax(output, dim=1).detach().numpy()
            # num = 1
            if(count==0):
                f[count*1:1,:] = output
            else:
                f[count*1:(count+1)*1,:] = output
            count+=1
        print(count)
        np.savetxt(out_feature_dir+'/features_te.txt', f[:count, :])
    else:
        f = np.zeros((len(test_loader) * size, 200))
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(async=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
            if (flag == 't'):
                output = model.forward_txt(input_var)
            else:
                output = model.forward_share(input_var)
            # if(flag == 'v'):
                # output = torch.mean(output,0).reshape(1,200)#video frame average
            output = F.softmax(output, dim=1).detach().cpu().numpy()
            num += output.shape[0]
            if (i == len(test_loader) - 1):
                f[i * size:num, :] = output
            else:
                f[i * size:(i + 1) * size, :] = output

        np.savetxt(out_feature_dir + '/features_te.txt', f[:num, :])
    return out_feature_dir + '/features_te.txt'
def extra_i(model,model_w,test_loader, out_feature_dir, args, flag):
    size = args.batch_size
    num = 0
    # if(flag == 'v'):
    #     size = 1
    #     f = np.zeros((len(test_loader),200))
    # else:
    f = np.zeros((len(test_loader) * size, 200))
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
        output_res = model.forward_share(input_var)
        output_w = model_w.forward(input_var)
        if (flag == 'i'):
            a=0.45
        if (flag == 'a'):
            a=0.6
        output = a * output_res + (1 - a) * output_w
        # if(flag == 'v'):
            # output = torch.mean(output,0).reshape(1,200)#video frame average
        output = F.softmax(output, dim=1).detach().cpu().numpy()
        num += output.shape[0]
        if (i == len(test_loader) - 1):
            f[i * size:num, :] = output
        else:
            f[i * size:(i + 1) * size, :] = output

    np.savetxt(out_feature_dir + '/features_te.txt', f[:num, :])
    return out_feature_dir + '/features_te.txt'
def extra_t(model,model_w,test_loader,test_loader_w, out_feature_dir, args, flag):
    size = args.batch_size
    num = 0
    # if(flag == 'v'):
    #     size = 1
    #     f = np.zeros((len(test_loader),200))
    # else:
    f = np.zeros((len(test_loader) * size, 200))
    for (i, (input, target)), (j, (input1, target1)) in zip(enumerate(test_loader), enumerate(test_loader_w)):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            input_var1 = Variable(input1).cuda()
        output_res = model.forward_txt(input_var)
        output_w = model_w.forward(input_var1)
        a=0.5
        output = a * output_res + (1 - a) * output_w
        # if(flag == 'v'):
            # output = torch.mean(output,0).reshape(1,200)#video frame average
        output = F.softmax(output, dim=1).detach().cpu().numpy()
        num += output.shape[0]
        if (i == len(test_loader) - 1):
            f[i * size:num, :] = output
        else:
            f[i * size:(i + 1) * size, :] = output

    np.savetxt(out_feature_dir + '/features_te.txt', f[:num, :])
    return out_feature_dir + '/features_te.txt'
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
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return test_loader


def get_test_set_v(data_dir, test_list, args):
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
    test_set = CubDataset_v(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return test_loader


def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset1(data_dir, test_list, split)
    data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return data_loader


if __name__ == "__main__":
    main()
