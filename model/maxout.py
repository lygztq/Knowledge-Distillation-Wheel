from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


def maxout(inputs, num_units, axis=-1, scope=None):  
    """
    Adds a maxout op from https://arxiv.org/abs/1302.4389

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, Yoshua Bengio

    Usually the operation is performed in the filter/channel dimension. This can also be used after 
    fully-connected layers to reduce number of features.

    Arguments:
    - inputs:     Tensor input

    - num_units:  Specifies how many features will remain after maxout in the `axis` dimension (usually channel).
                This must be multiple of number of `axis`.

    - axis:       The dimension where max pooling will be performed. Default is the last dimension.

    - scope:      Optional scope for variable_scope.

    Returns:
    A `Tensor` representing the results of the pooling operation.

    Raises:
    ValueError: if num_units is not multiple of number of features.
    """
    with variable_scope.variable_scope(scope, 'MaxOut', [inputs]):
        inputs = ops.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('Number of features({}) is not '
                            'a multiple of num_units({})'.format(
                                num_channels, num_units
                            ))
        shape[axis] = num_units
        shape += [num_channels // num_units]

        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = array_ops.shape(inputs)[i]
        outputs = math_ops.reduce_max(
            array_ops.reshape(inputs, shape), -1, keepdims=False
        )
        return outputs