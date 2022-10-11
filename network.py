import mindspore

from operations import ResNetBasicblock
from cell import InferCell
import mindspore.numpy as np
from mindspore import dtype as mstype
from mindspore import ops, Tensor, context, nn
from mindspore.common.api import ms_function

# def _AvgPool1_1(input):
#     # print(numpy.arange(1, dtype=mindspore.float32) * input.size(-1))
#     # s_p1 = (numpy.arange(1, dtype=mindspore.float32) * input.size(-1)).long()
#     # e_p1 = ((numpy.arange(1, dtype=mindspore.float32) + 1) * input.size(-1)).long()
#     #
#     # s_p2 = (numpy.arange(1, dtype=mindspore.float32) * input.size(-2)).long()
#     # e_p2 = ((numpy.arange(1, dtype=mindspore.float32) + 1) * input.size(-2)).long()
#
#     pooled2 = []
#     pooled = []
#     reduceMean_ = ops.ReduceMean(keep_dims=True)
#     print('xiaotian', reduceMean_)
#     res = reduceMean_(input[:, :, 0:input.size(-2), 0:input.size(-1)], (-2, -1))
#     # pooled.append(res)
#     # pooled2.append(pooled)
#     return res

class AdaptiveAvgPool2d_(nn.Cell):
    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool2d."""
        super(AdaptiveAvgPool2d_, self).__init__()
        self.output_size = output_size

    def adaptive_avgpool2d(self, inputs):
        """ NCHW """
        H = self.output_size[0]
        W = self.output_size[1]

        H_start = ops.Cast()(np.arange(start=0, stop=H, dtype=mstype.float32) * (inputs.shape[-2] / H), mstype.int64)
        H_end = ops.Cast()(np.ceil(((np.arange(start=0, stop=H, dtype=mstype.float32)+1) * (inputs.shape[-2] / H))), mstype.int64)

        W_start = ops.Cast()(np.arange(start=0, stop=W, dtype=mstype.float32) * (inputs.shape[-1] / W), mstype.int64)
        W_end = ops.Cast()(np.ceil(((np.arange(start=0, stop=W, dtype=mstype.float32)+1) * (inputs.shape[-1] / W))), mstype.int64)

        pooled2 = []
        for idx_H in range(H):
            pooled1 = []
            for idx_W in range(W):
                h_s = int(H_start[idx_H].asnumpy())
                h_e = int(H_end[idx_H].asnumpy())
                w_s = int(W_start[idx_W].asnumpy())
                w_e = int(W_end[idx_W].asnumpy())
                res = inputs[:, :, h_s:h_e, w_s:w_e]
                # res = inputs[:, :, H_start[idx_H]:H_end[idx_H], W_start[idx_W]:W_end[idx_W]]  # 这样写mindspore tensor切片报类型错误，不知道为啥
                pooled1.append(ops.ReduceMean(keep_dims=True)(res, (-2,-1)))
            pooled1 = ops.Concat(-1)(pooled1)
            pooled2.append(pooled1)
        pooled2 = ops.Concat(-2)(pooled2)

        return pooled2

    def construct(self, x):
        x = self.adaptive_avgpool2d(x)
        return x


class TinyNetwork(nn.Cell):
    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.SequentialCell(
            nn.Conv2d(in_channels=3, out_channels=C, kernel_size=3, padding=1, pad_mode="pad", has_bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.work_cells = nn.CellList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.work_cells.append(cell)
            # self.cells = nn.CellList([cell])
            C_prev = cell.out_dim
            # print(self.cells)
        # self._Layer = len(self.cells

        self.lastact = nn.SequentialCell(nn.BatchNorm2d(C_prev), nn.ReLU()) # delete inplace
        self.global_pooling = AdaptiveAvgPool2d_(output_size=(1,1))
        self.classifier = nn.Dense(in_channels=C_prev, out_channels=num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.work_cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.work_cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def construct(self, inputs):
        feature = self.stem(inputs)
        for cell in self.work_cells:
            feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.shape[0], -1)
        logits = self.classifier(out)

        return logits
