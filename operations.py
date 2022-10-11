from mindspore import nn, ops
from mindspore.common.api import ms_function

OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(
        C_in, C_out, stride
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1, 1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0, 0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}


class ReLUConvBN(nn.Cell):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                pad_mode="pad",
                dilation=dilation,
                has_bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine
            ),
        )

    def construct(self, x):
        return self.op(x)


class ResNetBasicblock(nn.Cell):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.SequentialCell(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, pad_mode="valid", has_bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def construct(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Cell):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def construct(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Zero(nn.Cell):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def construct(self, x):
        if self.C_in == self.C_out:
            mul = ops.Mul()
            if self.stride == 1:
                return mul(x, 0.0)
            else:
                return mul(x[:, :, :: self.stride, :: self.stride], 0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Cell):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU()
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.CellList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, pad_mode="valid", has_bias=not affine
                    )
                )
            self.pad = ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, pad_mode="valid", has_bias=not affine
            )
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine
        )

    def construct(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            concat_op = ops.Concat(axis=1)
            ## TODO: In MindSpore，converting low precision to high precision is needed before concat.
            # out = ops.Concat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
            out = concat_op(self.convs[0](x), self.convs[1](y[:, :, 1:, 1:]))
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        div = ops.div()
        x = div(x, keep_prob)
        mul = ops.mul
        x = mul(x, mask)
    return

def _check(input_shape, padding):
    """
    Check relationship between input shape and padding to make sure after negative dimension padding the out is
    positive.
    """
    if len(input_shape) < len(padding):
        msg = 'Dimension of input must more than or equal to len(padding)/2'  # modify
        raise ValueError(msg)
    if len(input_shape) > len(padding):
        if len(padding) == 2 and isinstance(padding[0], int):
            padding = [(0, 0) for i in range(len(input_shape) - 1)] + [padding]
        else:
            padding = [(0, 0) for i in range(len(input_shape) - len(padding))] + [x for x in padding]
    for index, item in enumerate(padding):
        if item[0] < -input_shape[index]:
            msg = 'Dimension out of range, expected no less than -{}, but got {}'.format(input_shape[index],
                                                                                         item[0])
            raise ValueError(msg)
        if item[1] < -input_shape[index]:
            msg = 'Dimension out of range, expected no less than -{}, but got {}'.format(input_shape[index],
                                                                                         item[1])
            raise ValueError(msg)
        if input_shape[index] + item[0] + item[1] <= 0:
            msg = 'The input size {}, plus negative padding {} and {} resulted in a non-positive output size, ' \
                  'which is invalid. Check dimension of your input'.format(input_shape[index], item[0], item[1])
            raise ValueError(msg)
    return padding


def _get_new_padding(padding):
    """get non-negative padding and make negative position."""
    new_padding = [[item[0], item[1]] for item in padding]
    start = [0 for i in range(len(new_padding))]
    end = [0 for i in range(len(new_padding))]
    for index, item in enumerate(new_padding):
        if item[0] < 0:
            start[index] = item[0]
            new_padding[index][0] = 0
        if item[1] < 0:
            end[index] = item[1]
            new_padding[index][1] = 0
    new_padding = tuple(new_padding)
    return new_padding, start, end


def _get_begin_size(shape, begin, end):
    """Calculate begin and size for ops.Slice."""
    size = tuple([shape[i] + begin[i] + end[i] for i in range(len(shape))])
    begin = tuple([int(-i) for i in begin])
    return begin, size

class _ConstantPadNd(nn.Cell):
    r"""
    Using a given value to pads the last n dimensions of input tensor.

    Args:
        padding(tuple, list): The padding size to pad the last n dimensions of input tensor. The padding
            sequence is starting from the last dimension and moving forward. The length of padding must be
            a multiple of 2. len(padding)/2 dimensions of input will be padded.
        value(union[int, float]): Padding value.

         padding (union[list, tuple]): The padding size to pad the last n dimensions of input tensor.
            The padding sequence is starting from the last dimension and moving forward.
            The length of padding must be a multiple of 2. If padding is :math:`(padding_0, padding_1, padding_2,
            padding_3, ..., padding_2m, padding_{2m+1}, ...)`. The input is `x`,
            the size of last dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`.
            The size of penultimate dimension of output is :math:`padding\_2 + x.shape[-2] + padding\_3`.
            The size of 3rd to last dimension of output is :math:`padding\_4 + x.shape[-3] + padding\_5`.
            The size of i-td to last dimension of output is :math:`padding\_{2m} + x.shape[-m-1] + padding\_{2m+1}`.
            The remaining dimensions of the output are consistent with those of the input.
        value (union[int, float]): Padding value.

    Returns:
        Tensor, the tensor after padding.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        ValueError: If the length of padding is not a multiple of 2.
        ValueError: If the length of input less than len(padding)/2.
        ValueError: If the output shape after padding is not positive.
    """

    def __init__(self, padding, value):
        """Initialize Pad."""
        super(_ConstantPadNd, self).__init__()
        self.value = value
        self.padding = self._to_ms_padding(padding)

    def _to_ms_padding(self, padding):
        """Transform the padding to the format of ms.nn.Pad."""
        if len(padding) % 2 != 0:
            msg = 'the length of padding must be a multiple of 2.'
            raise ValueError(msg)
        new_padding = []
        for i in range(len(padding) // 2):
            new_padding.append([padding[2 * i], padding[2 * i + 1]])
        new_padding.reverse()
        return new_padding

    def construct(self, x):
        """Construct the pad net."""
        input_shape = x.shape
        input_type = x.dtype
        padding = _check(input_shape, self.padding)
        new_padding, start, end = _get_new_padding(padding)
        mask = ops.Ones()(input_shape, input_type)
        output = ops.Pad(new_padding)(x)
        mask = ops.Pad(new_padding)(mask)
        ones = ops.Ones()(output.shape, output.dtype)
        value = ops.Fill()(output.dtype, output.shape, self.value)
        output = ops.Add()(ops.Mul()(mask, output), ops.Mul()(ops.Sub()(ones, mask), value))
        slice_op = ops.Slice()
        begin, size = _get_begin_size(output.shape, start, end)
        output = slice_op(output, begin, size)
        return output


class ConstantPad2d(_ConstantPadNd):
    r"""
    Using a given constant value to pads the last two dimensions of input tensor.

    Args:
        padding (union[int, tuple]): The padding size to pad the last two dimensions of input tensor.
            If is int, uses the same padding in boundaries of input's last two dimensions.
            If is tuple and length of padding is 4 uses (padding_0, padding_1, padding_2, padding_3) to pad.
            If the input is `x`, the size of last dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`.
            The size of penultimate dimension of output is :math:`padding\_2 + x.shape[-2] + padding\_3`.
            The remaining dimensions of the output are consistent with those of the input.
        value (union[int, float]): Padding value.

    Returns:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not a tuple or int.
        TypeError: If `value` is not int or float.
        ValueError: If the length of `padding` is more than 4 or not a multiple of 2.
        ValueError: If the output shape after padding is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        # >>> import numpy as np
        # >>> from mindspore import Tensor
        # >>> from mindspore.nn import ConstantPad2d
        # >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        # >>> x = Tensor(x)
        # >>> padding = (-1, 1, 0, 1)
        # >>> value = 0.5
        # >>> pad2d = ConstantPad2d(padding, value)
        # >>> out = pad2d(x)
        # >>> print(out)
        [[[[1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5]]
          [[1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5]]]]
        # >>> print(out.shape)
        (1, 2, 4, 4)
    """

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) // 2 > 2:
                msg = 'the length of padding with tuple type must less than or equal to 4.'
                raise ValueError(msg)
        else:
            msg = 'type of padding must be int or float, but got {}'.format(type(padding))
            raise TypeError(msg)

        if not isinstance(value, (int, float)):
            msg = 'type of value must be int or float, but got {}'.format(type(value))
            raise TypeError(msg)
        super(ConstantPad2d, self).__init__(padding, value)