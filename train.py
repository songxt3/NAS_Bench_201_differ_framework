import argparse
import random
import utils
import torch
import torch.nn as nn
import torchvision.datasets as dset
import time
import os

from network import TinyNetwork
from genotypes import Structure


parser = argparse.ArgumentParser("CIFAR")
#parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('using device {}'.format(device))


def train(train_queue, model, criterion, optimizer, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, top5.avg, objs.avg


def main():
    '''
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    '''
    train_transform, valid_transform = utils._data_transforms_cifar10()
    train_data = dset.CIFAR10(root='dataset/', train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root='dataset/', train=False, download=True, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=2)

    op = ['nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none']
    for id in range(1):
        code = []
        for _ in range(6):
            code.append(random.randint(0, 4))

        genotype = Structure(
            [
                ((op[code[0]], 0),),
                ((op[code[1]], 0), (op[code[2]], 1)),
                ((op[code[3]], 0), (op[code[4]], 1), (op[code[5]], 2))
            ]
        )
        print('id:%d, code:%s'%(id, str(code)))
        # mox.file.append('s3://nas-software-engineering/train_xt_201/main.log', 'id:%d, code:%s'%(id, str(code)) + '\n')
        model = TinyNetwork(C=16, N=5, genotype=genotype, num_classes=10)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        criterion = criterion.cuda()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate_min)

        best_top1 = 0.0
        info_str = ''
        for epoch in range(args.epochs):
            start_time = time.time()

            lr = scheduler.get_last_lr()

            train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
            valid_top1, valid_top5, valid_obj = infer(valid_queue, model, criterion)
            info = 'epoch:%d, lr:%.5f, train_acc:%.5f, valid_acc:%.5f'%(epoch, lr[0], train_acc, valid_top1) + '\n'
            info_str += info
            print(info)

            if valid_top1 > best_top1:
                best_top1 = valid_top1

            scheduler.step()

            end_time = time.time()
            print('epoch %d time:%s'%(int(epoch), str(end_time - start_time)))

        print('best_acc:%.5f'%best_top1)


if __name__ == '__main__':
    main()