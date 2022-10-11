import os
import numpy as np
# import torch
import shutil
# import torchvision.transforms as transforms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_transforms_1
import mindspore.dataset.transforms.c_transforms as c_transforms_2
from mindspore import dtype as mstype
from mindspore.communication.management import get_rank, get_group_size
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def get_train_valid_loader(data_dir, batch_size, training=True, num_workers=8, repeat_num=1):
    if training:
        usage = 'train'
    else:
        usage = 'test'

    normalize = c_transforms_1.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    type_cast_op = c_transforms_2.TypeCast(mstype.int32)

    if training:
        trans = c_transforms_2.Compose([
            c_transforms_1.RandomCrop(32, padding=4),
            c_transforms_1.RandomHorizontalFlip(),
            normalize,
            c_transforms_1.HWC2CHW(),
            # c_transforms_1.CutOut(16)
        ])
    else:
        trans = c_transforms_2.Compose([
            normalize,
            c_transforms_1.HWC2CHW()
        ])

    # load the datase
    cifar_ds = ds.Cifar10Dataset(dataset_dir=data_dir, usage=usage, num_parallel_workers=num_workers)
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=trans, input_columns="image")

    data_set = cifar_ds.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


# def create_dataset1(dataset_path, do_train, batch_size=32, train_image_size=224, eval_image_size=224,
#                     target="Ascend", distribute=False, enable_cache=False, cache_session_id=None):
#     """
#     create a train or evaluate cifar10 dataset for resnet50
#     Args:
#         dataset_path(string): the path of dataset.
#         do_train(bool): whether dataset is used for train or eval.
#         repeat_num(int): the repeat times of dataset. Default: 1
#         batch_size(int): the batch size of dataset. Default: 32
#         target(str): the device target. Default: Ascend
#         distribute(bool): data for distribute or not. Default: False
#         enable_cache(bool): whether tensor caching service is used for eval. Default: False
#         cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None
#
#     Returns:
#         dataset
#     """
#     device_num, rank_id = _get_rank_info(distribute)
#     ds.config.set_prefetch_size(64)
#     if device_num == 1:
#         data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=get_num_parallel_workers(12), shuffle=True)
#     else:
#         data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=get_num_parallel_workers(12), shuffle=True,
#                                      num_shards=device_num, shard_id=rank_id)
#
#     # define map operations
#     trans = []
#     if do_train:
#         trans += [
#             ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
#             ds.vision.RandomHorizontalFlip(prob=0.5)
#         ]
#
#     trans += [
#         ds.vision.Resize((train_image_size, train_image_size)),
#         ds.vision.Rescale(1.0 / 255.0, 0.0),
#         ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
#         ds.vision.HWC2CHW()
#     ]
#
#     type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)
#
#     data_set = data_set.map(operations=type_cast_op, input_columns="label",
#                             num_parallel_workers=get_num_parallel_workers(8))
#     # only enable cache for eval
#     if do_train:
#         enable_cache = False
#     if enable_cache:
#         if not cache_session_id:
#             raise ValueError("A cache session_id must be provided to use cache.")
#         eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
#         data_set = data_set.map(operations=trans, input_columns="image",
#                                 num_parallel_workers=get_num_parallel_workers(8), cache=eval_cache)
#     else:
#         data_set = data_set.map(operations=trans, input_columns="image",
#                                 num_parallel_workers=get_num_parallel_workers(8))
#
#     # apply batch operations
#     data_set = data_set.batch(batch_size, drop_remainder=True)
#
#     return data_set


def create_dataset_cifar10(cfg, data_path, batch_size=32, status="train", target="Ascend",
                           num_parallel_workers=8):
    """
    create dataset for train or test
    """

    ds.config.set_prefetch_size(64)
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()

    if target != "Ascend" or device_num == 1:
        cifar_ds = ds.Cifar10Dataset(data_path, shuffle=True)
    else:
        cifar_ds = ds.Cifar10Dataset(data_path, num_parallel_workers=num_parallel_workers,
                                     shuffle=True, num_shards=device_num, shard_id=rank_id)
    rescale = 1.0 / 255.0
    shift = 0.0
    # cfg = alexnet_cifar10_cfg

    resize_op = CV.Resize((cfg.image_height, cfg.image_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if status == "train":
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op,
                            num_parallel_workers=1)
    if status == "train":
        compose_op = [random_crop_op, random_horizontal_op, resize_op, rescale_op, normalize_op, channel_swap_op]
    else:
        compose_op = [resize_op, rescale_op, normalize_op, channel_swap_op]
    cifar_ds = cifar_ds.map(input_columns="image", operations=compose_op, num_parallel_workers=num_parallel_workers)

    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    return cifar_ds


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


# def save_checkpoint(state, is_best, save):
#   filename = os.path.join(save, 'checkpoint.pth.tar')
#   torch.save(state, filename)
#   if is_best:
#     best_filename = os.path.join(save, 'model_best.pth.tar')
#     shutil.copyfile(filename, best_filename)


# def save(model, model_path):
#   torch.save(model.state_dict(), model_path)


# def load(model, model_path):
#   model.load_state_dict(torch.load(model_path))


# def drop_path(x, drop_prob):
#   if drop_prob > 0.:
#     keep_prob = 1.-drop_prob
#     mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
#     x.div_(keep_prob)
#     x.mul_(mask)
#   return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
