import numpy as np
from methods import align_and_pad, derivatives, ReLU, glorot_uniform, zeros 
from build import cmethods

### Allocating Layer ###

# Store 2D | 3D | 4D tensor 
class Input():
    def __init__(self, a):
        self.previous = None
        self.a = a
        self.z = None
        self.df = None

# 4D | 3D | 2D tensor --> 4D | 3D | 2D tensor
class Reshape():
    def __init__(self, shape=-1):
        self.previous = None
        self.shape = shape
        self.original_shape = None
        self.df = None
        self.a = None
        self.z = None
    
    # 3D | 2D | 1D tensor --> 3D | 2D | 1D tensor
    def forward_pass(self, previous=None):
        self.previous = previous
        self.original_shape = previous.a.shape
        self.df = previous.df
        self.a = previous.a.reshape(previous.a.shape[0], *self.shape)
        self.z = previous.z.reshape(previous.z.shape[0], *self.shape)
        return self.a

    # 3D | 2D | 1D tensor <-- 3D | 2D | 1D tensor
    def backward_pass(self, dz_next):
        return dz_next.reshape(dz_next.shape[0], *self.original_shape)

# 4D | 3D tensor --> 2D tensor
class Flatten():
    def __init__(self):
        self.previous = None
        self.original_shape = None
        self.df = None
        self.n = 0
        self.a = None
        self.z = None
    
    # 3D | 2D tensor --> 1D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.original_shape = previous.a[0].shape
        self.df = previous.df
        self.a = previous.a.reshape(previous.a.shape[0], -1)
        self.z = previous.z.reshape(previous.z.shape[0], -1)
        self.n = self.a[0].size
        return self.a
    
    # 3D | 2D tensor <-- 1D tensor
    def backward_pass(self, dz_next):
        return dz_next.reshape(dz_next.shape[0], *self.original_shape)


### Trainable Layers ###

# 2D tensor --> 2D tensor
class Dense():
    def __init__(self, n=0, activation=None, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n = n
        self.previous = None
        self.a = None
        self.w = None
        self.b = None
        self.z = None
        self.f = activation
        self.df = derivatives[activation]
        self.f_w = initializer_w
        self.f_b = initializer_b

    # 1D tensor --> 1D tensor
    def forward_pass(self, previous):
        self.previous = previous

        # initialize weights and baises
        if self.w is None:
            # register these new pramaters
            self.w, self.b = self.f_w(self.previous.n, self.n), self.f_b((self.n,))
        
        # calculate activations
        self.z = previous.a @ self.w.T + self.b
        self.a = self.f(self.z)
        return self.a

    def calculate_gradient(self, dz):
        # calculate the loss gradient with respect to 
        # the weights and the biases for dense layer
        # dw = np.outer(dz, self.previous.a)
        #  -------dz--------
        # |
        # |
        # a
        # |
        # |
        dW = dz.T @ self.previous.a
        dB = np.sum(dz, axis=0)
        return dW, dB

    # 1D tensor <-- 1D tensor
    def backward_pass(self, dz):
        # da = self.w.T @ dz
        da = dz @ self.w
        return da * self.previous.df(self.previous.z)

# 4D tensor --> 4D tensor
class Conv():
    def __init__(self, n_kernels=1, k_shape=(1, 1), input_depth=None, stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n_kernels = n_kernels
        self.k_shape = k_shape
        self.depth = input_depth
        self.stride = stride
        self.pad = padding

        # store the previous element to form a hierarchy 
        self.previous = None

        # functions
        self.f = activation
        self.df = derivatives.get(activation)
        self.f_w = initializer_w
        self.f_b = initializer_b
                
        # parameters
        self.w = None
        self.b = None
        self.z = None
        self.a = None
        self.padded_input = None

    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        # use the depth of the previous activations as input depth
        if not self.depth: self.depth = previous.a.shape[1]
        
        # initialize weights
        if self.w is None:
            self.w = np.array(
                [
                    [self.f_w(*self.k_shape) for _ in range(self.depth)]
                    for _ in range(self.n_kernels)
                ],
                dtype=float
            )

        height = self.previous.a.shape[2] + 2 * self.pad
        width = self.previous.a.shape[3] + 2 * self.pad
        
        # This corrects the input boundaries to align with kernel shape,
        # so that the kernel will cover all the input with the least padding 
        # room possible.
        (u, d), (l, r) = align_and_pad((height, width), self.w[0][0].shape, self.stride)
        pad_width = (0, 0), (0, 0), (u + self.pad, d + self.pad), (l + self.pad, r + self.pad)

        self.padded_input = np.pad(self.previous.a, pad_width, mode='constant')

        out = cmethods.cross_corr4D(self.padded_input, self.w, self.stride)
        
        # initialize baises
        if self.b is None:
            self.b = self.f_b(out.shape[1:])
        
        self.z = self.b + out
        self.a = self.f(self.z)
        
        return self.a

    def calculate_gradient(self, dz):
        # calculate the loss gradient with respect to the
        # weights and the biases for convolutional layer
        dw = cmethods.cross_corrTransposed4D(dz, self.padded_input, *self.k_shape, self.stride)     
        db = np.sum(dz, axis=0)
        return dw, db

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        da = cmethods.convTransposed4D(dz_next, self.w, *self.previous.a.shape[2:], self.stride) # full padding
        return da * self.previous.df(self.previous.z)


### Pooling Layers ###

# 4D tensor --> 4D tensor
class MaxPool():
    def __init__(self, size=(1, 1), stride=1):
        self.size = size
        self.stride = stride
        self.previous = None

        # activations
        self.df = None # copied from the previous layer
        
        # parameters
        self.mask = None # to distribute the gradient based on each input contribution 
        self.a = None
        self.z = None

    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        self.a, self.z, self.mask = cmethods.MaxPooling4D(previous.a, previous.z, *self.size, self.stride)
        return self.a

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        return cmethods.MaxMinPoolingTransposed4D(dz_next, self.mask, *self.size, self.stride)

# 4D tensor --> 4D tensor
class MinPool(MaxPool):
    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        self.a, self.z, self.mask = cmethods.MinPooling4D(previous.a, previous.z, *self.size, self.stride)
        return self.a

# 4D tensor --> 4D tensor
class AveragePool(MaxPool):
    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        self.a, self.z, self.mask = cmethods.AveragePooling4D(previous.a, previous.z, *self.size, self.stride)
        return self.a

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        return cmethods.AveragePoolingTransposed4D(dz_next, *self.previous.a.shape[2:], *self.size, self.stride)

# 4D tensor --> 4D tensor (the last dimension is single-elmenet array)
class GlobalMaxPool():
    def __init__(self, stride=1):
        self.stride = stride
        self.previous = None

        # activations
        self.df = None # copied from the previous layer
        
        # parameters
        self.mask = None # to distribute the gradient based on each input contribution 
        self.a = None
        self.z = None

    # 3D tensor --> 1D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        self.a, self.z, self.mask = cmethods.MaxPooling4D(previous.a, previous.z, *previous.a.shape[2:], self.stride)
        return self.a

    # 3D tensor <-- 1D tensor
    def backward_pass(self, dz_next):
        return cmethods.MaxMinPoolingTransposed4D(dz_next, self.mask, *self.previous.a.shape[2:], self.stride)

# 4D tensor --> 4D tensor (the last dimension is single-elmenet array)
class GlobalMinPool(GlobalMaxPool):
    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        self.a, self.z, self.mask = cmethods.MinPooling4D(previous.a, previous.z, *previous.a.shape[2:], self.stride)
        return self.a

# 4D tensor --> 4D tensor (the last dimension is single-elmenet array)
class GlobalAveragePool(GlobalMaxPool):
    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        self.a, self.z = cmethods.AveragePooling4D(previous.a, previous.z, *previous.a.shape[2:], self.stride)
        return self.a

    def backward_pass(self, dz_next):
        size = self.previous.a.shape[2:]
        return cmethods.AveragePoolingTransposed4D(dz_next, *size, *size, self.stride)

# 4D tensor --> 4D tensor
class AdaptiveMaxPool(MaxPool):
    def __init__(self, output_size=(1, 1)):
        self.output_size = output_size
        super().__init__()

    def forward_pass(self, previous):
        self.size = (previous.a.shape[0] - self.output_size[0] + 1, previous.a.shape[1] - self.output_size[1] + 1)
        return super().forward_pass(previous)

# 4D tensor --> 4D tensor
class AdaptiveMinPool(MinPool):
    def __init__(self, output_size=(1, 1)):
        self.output_size = output_size
        super().__init__()

    def forward_pass(self, previous):
        self.size = (previous.a.shape[0] - self.output_size[0] + 1, previous.a.shape[1] - self.output_size[1] + 1)
        return super().forward_pass(previous)

# 4D tensor --> 4D tensor
class AdaptiveAveragePool(AveragePool):
    def __init__(self, output_size=(1, 1)):
        self.output_size = output_size
        super().__init__()

    def forward_pass(self, previous):
        self.size = (previous.a.shape[0] - self.output_size[0] + 1, previous.a.shape[1] - self.output_size[1] + 1)
        return super().forward_pass(previous)
