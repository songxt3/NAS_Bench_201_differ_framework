import argparse


import utils
from mindspore import nn, context, save_checkpoint, Tensor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.profiler import Profiler
import time

from network import TinyNetwork
from genotypes import Structure
import local_adapter
import os
import stat
import numpy as np
import random


parser = argparse.ArgumentParser("CIFAR")
#parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--data_dir', type=str, default='dataset/', help='data dir')
parser.add_argument('--data_url', type=str, default=None, help='obs path of dataset')
parser.add_argument('--train_url', type=str, default=None, help='obs path of training result')
args = parser.parse_args()


class Eval_callback(Callback):
    def __init__(self, eval_function, eval_param_dict, interval=1, save_best_ckpt=True,
                 ckpt_directory="./cache/", besk_ckpt_name="best0.ckpt", metrics_name="acc"):
        # be changed
        super(Eval_callback, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.epochs = args.epochs
        self.eval_start_epoch = 1
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss_epoch = cb_params.net_outputs
        if cur_epoch >= self.eval_start_epoch:
            res = self.eval_function(self.eval_param_dict)
            print('-' * 10)
            print('Epoch {}/{}'.format(cur_epoch, self.epochs))
            print('train Loss: {}'.format(loss_epoch))
            print('val Acc: {}'.format(res))
            write()
            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)

    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch))

def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    metrics_name = eval_param['metrics_name']
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

def get_lr_cifar10(current_step, lr_max, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       current_step(int): current steps of the training
       lr_max(float): max learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        else:
            lr = lr_max * 0.1
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate

def main():
    '''
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    '''
    train_queue = utils.get_train_valid_loader(data_dir=args.data_dir, batch_size=args.batch_size, training=True)
    valid_queue = utils.get_train_valid_loader(data_dir=args.data_dir, batch_size=args.batch_size, training=False)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    context.set_context(save_graphs=False)
    print('device id:', local_adapter.get_device_id())
    context.set_context(device_id=0)

    op = ['nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none']
    for id in range(2500): # only test one structure
        start_time = time.time()

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
        print('id:', id, 'code:', code)
        network = TinyNetwork(C=16, N=5, genotype=genotype, num_classes=10)
        network.add_flags_recursive(fp16=True)

        profiler = Profiler(output_path='./profiler_data')

        step_per_epoch = train_queue.get_dataset_size()
        min_lr = args.learning_rate_min
        max_lr = args.learning_rate
        total_step = step_per_epoch * args.epochs
        lr = Tensor(nn.cosine_decay_lr(max_lr=max_lr, min_lr=min_lr, total_step=total_step, step_per_epoch=step_per_epoch, decay_epoch=args.epochs))
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = nn.SGD(
            params=network.trainable_params(),
            learning_rate=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )

        metrics = {"Accuracy": Accuracy()}

        model = Model(network, loss_fn=criterion, optimizer=optimizer, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False)

        time_cb = TimeMonitor(data_size=step_per_epoch)
        config_ck = CheckpointConfig(save_checkpoint_steps=1562, keep_checkpoint_max=10)
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_test_NASBench201", directory="cache/", config=config_ck)
        loss_cb = LossMonitor(per_print_times=50)
        eval_param_dict = {"model": model, "dataset": valid_queue, "metrics_name": "Accuracy"}
        eval_cb = Eval_callback(apply_eval, eval_param_dict, args.epochs)
        model.train(args.epochs, train_queue, callbacks=[eval_cb, time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
        end_time = time.time()
        duration_time = str(end_time - start_time)
        print('end_time:%s' % end_time)
        print('time:%s' % duration_time)
        profiler.analyse()


if __name__ == '__main__':
    main()