import os
import time
import torch
from torch.autograd import Variable
from util import AverageMeter, Log
from rankingloss import *


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)

    return loss  

def get_MMD(output,outputi,label_num, label_num_i):
    loss = torch.tensor(0.).cuda()
    for i in range(200):
        n=torch.min(label_num[i],label_num_i[i]).int()
        if n!=0:
            loss+=mmd_rbf(output[i][:n],outputi[i][:n])
    return loss

def train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion, center_loss, optimizer,
          epoch, num_epochs):
    print(len(train_loader), len(train_loader1), len(train_loader2), len(train_loader3))
    count = 0
    since = time.time()

    running_loss0 = AverageMeter()
    running_loss1 = AverageMeter()
    running_loss2 = AverageMeter()
    running_loss3 = AverageMeter()
    running_loss4 = AverageMeter()
    running_loss5 = AverageMeter()
    running_loss6 = AverageMeter()
    running_loss7 = AverageMeter()
    running_loss = AverageMeter()

    log = Log()
    model.train()

    image_acc = 0
    text_acc = 0
    video_acc = 0
    audio_acc = 0


    for (i, (input, target)), (j, (input1, target1)), (k, (input2, target2)), (p, (input3, target3)) in zip(
            enumerate(train_loader), enumerate(train_loader1), enumerate(train_loader2), enumerate(train_loader3)):
        input_var = Variable(input.cuda())
        input_var1 = Variable(input1.cuda())
        input_var2 = Variable(input2.cuda())
        input_var3 = Variable(input3.cuda())

        targets = torch.cat((target, target1, target2, target3), 0)
        targets = Variable(targets.cuda())

        target_var = Variable(target.cuda())
        target_var1 = Variable(target1.cuda())
        target_var2 = Variable(target2.cuda())
        target_var3 = Variable(target3.cuda())

        label_num_i = Variable(torch.zeros(200).cuda())
        label_num_v = Variable(torch.zeros(200).cuda())
        label_num_a = Variable(torch.zeros(200).cuda())
        label_num_t = Variable(torch.zeros(200).cuda())
        outputi = Variable(torch.zeros(200, len(input), 200).cuda())
        outputv = Variable(torch.zeros(200, len(input), 200).cuda())
        outputa = Variable(torch.zeros(200, len(input), 200).cuda())
        outputt = Variable(torch.zeros(200, len(input), 200).cuda())


        outputs = model(input_var, input_var1, input_var2, input_var3)  # [16,200]
        # print('output',outputs.size())
        size = int(outputs.size(0) / 4)
        img = outputs.narrow(0, 0, size)
        vid = outputs.narrow(0, size, size)
        aud = outputs.narrow(0, 2 * size, size)
        txt = outputs.narrow(0, 3 * size, size)

        for (i, j) in zip(target_var, img):
            outputi[i][label_num_i[i].int()]=j
            label_num_i[i] += 1
        for (i, j) in zip(target_var1, vid):
            outputv[i][label_num_i[i].int()] = j
            label_num_v[i] += 1
        for (i, j) in zip(target_var2, aud):
            outputa[i][label_num_i[i].int()] = j
            label_num_a[i] += 1
        for (i, j) in zip(target_var3, txt):
            outputt[i][label_num_i[i].int()] = j
            label_num_t[i] += 1
        # print(type(label_num_i[1]))


    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), "训练了%d个batch" % count)
        _, predict1 = torch.max(img, 1)  # 0是按列找，1是按行找
        _, predict2 = torch.max(vid, 1)  # 0是按列找，1是按行找
        _, predict3 = torch.max(aud, 1)  # 0是按列找，1是按行找
        _, predict4 = torch.max(txt, 1)  # 0是按列找，1是按行找
        image_acc += torch.sum(torch.squeeze(predict1.long() == target_var.long())).item() / float(target_var.size()[0])
        video_acc += torch.sum(torch.squeeze(predict2.long() == target_var1.long())).item() / float(
            target_var1.size()[0])
        audio_acc += torch.sum(torch.squeeze(predict3.long() == target_var2.long())).item() / float(
            target_var2.size()[0])
        text_acc += torch.sum(torch.squeeze(predict4.long() == target_var3.long())).item() / float(
            target_var3.size()[0])

        loss0 = criterion(img, target_var)
        loss1 = criterion(vid, target_var1)
        loss2 = criterion(aud, target_var2)
        loss3 = criterion(txt, target_var3)

        loss4 = loss0 + loss1 + loss2 + loss3
        loss5 = center_loss(outputs, targets) * 0.001
        # loss5=get_MMD(feature,targets)
        if (args.loss_choose == 'r'):
            ### loss6, _ = ranking_loss(targets, outputs, margin=1, margin2=0.5, squared=False)
            loss6, _ = ranking_loss(targets, feature, margin=1, margin2=0.5, squared=False)
            loss6 = loss6 * 0.1
        else:
            loss6 = 0.0

        ##loss = loss4 + loss5 + loss6
        loss7 = (get_MMD(outputv, outputi, label_num_v, label_num_i)+ get_MMD(outputa, outputi, label_num_a, label_num_i)+ get_MMD(outputt, outputi, label_num_t, label_num_i)\
        + get_MMD(outputa, outputv, label_num_a, label_num_v)+ get_MMD(outputt, outputv, label_num_t, label_num_v)+ get_MMD(outputa, outputt, label_num_a, label_num_t))*0.001
        # loss7 = get_MMD(outputv, outputi, label_num_v, label_num_i)
        loss = loss4+loss7
        # print(loss)
        batchsize = input_var.size(0)
        running_loss0.update(loss0.item(), batchsize)
        running_loss1.update(loss1.item(), batchsize)
        running_loss2.update(loss2.item(), batchsize)
        running_loss3.update(loss3.item(), batchsize)
        running_loss4.update(loss4.item(), batchsize)
        running_loss5.update(loss5.item(), batchsize)
        running_loss7.update(loss7.item(), batchsize)
        if (args.loss_choose == 'r'):
            running_loss6.update(loss6.item(), batchsize)
        running_loss.update(loss.item(), batchsize)
        optimizer.zero_grad()
        loss.backward()


        # for param in center_loss.parameters():
        #   param.grad.data *= (1./0.001)

        optimizer.step()
        count += 1
        if (i % args.print_freq == 0):

            print('-' * 20)
            print('Epoch [{0}/{1}][{2}/{3}]'.format(epoch, num_epochs, i, len(train_loader)))
            print('Image Loss: {loss.avg:.5f}'.format(loss=running_loss0))
            print('Video Loss: {loss.avg:.5f}'.format(loss=running_loss1))
            print('Audio Loss: {loss.avg:.5f}'.format(loss=running_loss2))
            print('Text Loss: {loss.avg:.5f}'.format(loss=running_loss3))
            print('AllMedia Loss: {loss.avg:.5f}'.format(loss=running_loss4))
            print('MMD Loss: {loss.avg:.5f}'.format(loss=running_loss7))
            if (args.loss_choose == 'r'):
                print('Ranking Loss: {loss.avg:.5f}'.format(loss=running_loss6))
            print('All Loss: {loss.avg:.5f}'.format(loss=running_loss))
            # log.save_train_info(epoch, i, len(train_loader), running_loss)


    # optimizer.zero_grad()
    # loss.backward()
    #
    # # for param in center_loss.parameters():
    # #   param.grad.data *= (1./0.001)
    #
    # optimizer.step()

    print("训练第%d个epoch:" % epoch)
    print("image:", image_acc / len(train_loader3))
    print("text:", text_acc / len(train_loader3))
    print("video:", video_acc / len(train_loader3))
    print("audio:", audio_acc / len(train_loader3))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), "训练了%d个batch" % count)


