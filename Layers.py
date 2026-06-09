import numpy as np
from methods import align_and_pad, derivatives, ReLU, glorot_uniform, zeros 
from build import cmethods
from time import perf_counter

### Allocating Layer ###

# Store 2D | 3D | 4D tensor 
class Input():
    def __init__(self, input_shape):
        self.previous = None
        self.out_shape = input_shape
        self.a = None
        self.z = None
        self.df = None

    def activate(self, a):
        self.a = a
    
    def update_batch_size(self, batch_size):
        self.out_shape = (
                batch_size,
                self.out_shape[1],
                self.out_shape[2],
                self.out_shape[3]
        )

# 4D | 3D | 2D tensor --> 4D | 3D | 2D tensor
class Reshape():
    def __init__(self, shape):
        self.previous = None
        self.shape = shape
        self.out_shape = None
        self.df = None
        self.a = None
        self.z = None
    
    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        self.out_shape = (
                previous.out_shape[0],
                self.shape[0],
                self.shape[1],
                self.shape[2]
        )

    def update_batch_size(self):
        self.out_shape = (
            self.previous.out_shape[0],
            self.shape[0],
            self.shape[1],
            self.shape[2]
        )

    # 3D | 2D | 1D tensor --> 3D | 2D | 1D tensor
    def forward_pass(self):
        self.a = self.previous.a.reshape(self.previous.out_shape[0], *self.out_shape)

        if self.previous.z is not None:
            self.z = self.previous.z.reshape(self.previous.out_shape[0], *self.out_shape)
        
        return self.a

    # 3D | 2D | 1D tensor <-- 3D | 2D | 1D tensor
    def backward_pass(self, dz):
        return dz.reshape(dz.shape[0], *self.previous.out_shape)

# 4D | 3D tensor --> 2D tensor
class Flatten():
    def __init__(self):
        self.previous = None
        self.out_shape = None
        self.df = None
        self.n = None
        self.a = None
        self.z = None

    def fuse(self, previous):
        self.previous = previous
        self.n = previous.out_shape[1] * previous.out_shape[2] * previous.out_shape[3]

        self.out_shape = (
            previous.out_shape[0], 
            self.n
        )

        self.df = previous.df

    def update_batch_size(self):
        self.out_shape = (
            self.previous.out_shape[0], 
            self.n
        )

    # 3D | 2D tensor --> 1D tensor
    def forward_pass(self):
        self.a = self.previous.a.reshape(self.previous.out_shape[0], -1)

        if self.previous.z is not None:
            self.z = self.previous.z.reshape(self.previous.out_shape[0], -1)
        
        return self.a
    
    # 3D | 2D tensor <-- 1D tensor
    def backward_pass(self, dz):
        return dz.reshape(dz.shape[0], *self.previous.out_shape[1:])


### Trainable Layers ###

# 2D tensor --> 2D tensor
class Dense():
    def __init__(self, n, activation=None, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n = n
        self.previous = None
        self.out_shape = None
        self.a = None
        self.w = None
        self.b = None
        self.z = None
        self.f = activation
        self.df = derivatives[activation]
        self.f_w = initializer_w
        self.f_b = initializer_b

    # 1D tensor --> 1D tensor
    def fuse(self, previous):    
        self.previous = previous
        self.out_shape = (previous.out_shape[0], self.n)
        
        # initialize weights
        if self.w is None:
            self.w = self.f_w(self.previous.n, self.n)
        
        # initialize baises
        if self.b is None:
            self.b = self.f_b((self.n,))

    def update_batch_size(self):
        self.out_shape = (self.previous.out_shape[0], self.n)

    def forward_pass(self):
        # calculate activations
        self.z = self.previous.a @ self.w.T + self.b
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
        dw = (dz.T @ self.previous.a) / self.out_shape[0]
        db = np.sum(dz, axis=0) / self.out_shape[0]
        return dw, db

    # 1D tensor <-- 1D tensor
    def backward_pass(self, dz):
        da = dz @ self.w
        return da * self.previous.df(self.previous.z)

# 4D tensor --> 4D tensor
class Conv():
    def __init__(self, n_kernels=1, k_shape=(1, 1), stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n_kernels = n_kernels
        self.k_shape = k_shape
        self.stride = stride
        self.pad = padding

        # store the previous element to form a hierarchy 
        self.previous = None
        self.out_shape = None

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
        self.pad_breadth = None
        self.padded_input = None
        self.padded_shape = None

        # math handlers
        self.x4d = None
        self.xT4d = None
        self.cT4d = None

        # algorithm
        self.feed_algorithm = {}
        self.grad_algorithm = {}
        self.back_algorithm = {}

    # 3D tensor --> 3D tensor
    def fuse(self, previous):

        self.previous = previous
    
        ''' calculate the output shape and the total padding '''
        
        height = self.previous.out_shape[2] + 2 * self.pad
        width = self.previous.out_shape[3] + 2 * self.pad
        
        # This corrects the input boundaries to align with kernel shape,
        # so that the kernel can cover all the input with the least padding 
        # room possible.
        (u, d), (l, r) = align_and_pad((height, width), self.k_shape, self.stride)
        self.pad_breadth = (0, 0), (0, 0), (u + self.pad, d + self.pad), (l + self.pad, r + self.pad)

        input_height = u + d + height
        input_width = l + r + width
        
        self.padded_shape = (
                    previous.out_shape[0],
                    previous.out_shape[1],
                    input_height,
                    input_width
        )
        
        self.out_shape = (
                    previous.out_shape[0],
                    self.n_kernels,
                    (input_height - self.k_shape[0]) // self.stride + 1, 
                    (input_width - self.k_shape[1])  // self.stride + 1
        )

        # initialize weights
        if self.w is None:
            self.w = np.array(
                [
                    [self.f_w(*self.k_shape) for _ in range(previous.out_shape[1])]
                    for _ in range(self.n_kernels)
                ],
                dtype=float
            )

        # initialize baises
        if self.b is None:
            self.b = self.f_b(self.out_shape[1:])

        # initialize C++ optimiztied methods
        self.x4d = cmethods.I2x_corr4D(*self.padded_shape, *self.w.shape, self.stride)
        self.xT4d = cmethods.I2x_corrT4D(*self.padded_shape, *self.out_shape, *self.k_shape, self.stride)
        self.cT4d = cmethods.I2x_convT4D(*self.w.shape, *self.out_shape, *self.previous.out_shape[2:], self.stride)
        
        if self.padded_shape not in self.feed_algorithm:
            self.automate_algorithm_choices() 

    def update_batch_size(self):
        batch_size = self.previous.out_shape[0]
        self.padded_shape = (
                batch_size,
                self.padded_shape[1],
                self.padded_shape[2],
                self.padded_shape[3]
        )

        self.out_shape = (
                batch_size,
                self.out_shape[1],
                self.out_shape[2],
                self.out_shape[3]
        )
        
        self.x4d.update(batch_size)
        self.xT4d.update(batch_size)
        self.cT4d.update(batch_size)

        if self.padded_shape not in self.feed_algorithm:
            self.automate_algorithm_choices()

    def automate_algorithm_choices(self):
        a_in = np.random.randn(*self.padded_shape)
        a_out = np.random.randn(*self.out_shape)

        # benchmark forward pass (optimized loop vs. optimized GEMM)
        t = perf_counter()
        a_out = self.x4d.loop(a_in, self.w)
        loop_elapsed_time = perf_counter() - t

        t = perf_counter()
        a_out = self.x4d.gemm(a_in, self.w)
        gemm_elapsed_time = perf_counter() - t

        if gemm_elapsed_time < loop_elapsed_time:
            self.feed_algorithm[self.padded_shape] = self.x4d.gemm 
        else:
            self.feed_algorithm[self.padded_shape] = self.x4d.loop 

        # benchmark gradient calculation (optimized loop vs. optimized GEMM)
        t = perf_counter()
        self.xT4d.loop(a_in, a_out)
        loop_elapsed_time = perf_counter() - t
        
        t = perf_counter()
        self.xT4d.gemm(a_in, a_out)
        gemm_elapsed_time = perf_counter() - t

        if gemm_elapsed_time < loop_elapsed_time:
            self.grad_algorithm[self.padded_shape] = self.xT4d.gemm
        else:
            self.grad_algorithm[self.padded_shape] = self.xT4d.loop 
        
        # benchmark backward pass (optimized loop vs. optimized GEMM)
        t = perf_counter()
        self.cT4d.loop(self.w, a_out)
        loop_elapsed_time = perf_counter() - t
        
        t = perf_counter()
        self.cT4d.gemm(self.w, a_out)
        gemm_elapsed_time = perf_counter() - t

        if gemm_elapsed_time < loop_elapsed_time:
            self.back_algorithm[self.padded_shape] = self.cT4d.gemm 
        else:
            self.back_algorithm[self.padded_shape] = self.cT4d.loop 

    def forward_pass(self):
        self.padded_input = np.pad(self.previous.a, self.pad_breadth, mode='constant')
        self.z = self.feed_algorithm[self.padded_shape](self.padded_input, self.w) + self.b
        self.a = self.f(self.z)

        return self.a

    def calculate_gradient(self, dz):
        # calculate the loss gradient with respect to the
        # weights and the biases for convolutional layer
        dw = self.grad_algorithm[self.padded_shape](self.padded_input, dz) / self.out_shape[0]
        db = np.sum(dz, axis=0) / self.out_shape[0]
        return dw, db

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz):
        da = self.back_algorithm[self.padded_shape](self.w, dz)
        return da * self.previous.df(self.previous.z)


### Pooling Layers ###

# 4D tensor --> 4D tensor
class MaxPool():
    def __init__(self, size=(1, 1), stride=1):
        self.size = size
        self.stride = stride
        self.previous = None
        self.out_shape = None

        # activations
        self.df = None # copied from the previous layer
        
        # parameters
        self.a = None
        self.z = None

        # math handler
        self.xp4D = None
    
    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df

        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            (previous.out_shape[2] - self.size[0]) // self.stride + 1,
            (previous.out_shape[3] - self.size[1]) // self.stride + 1
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, self.stride)
        
    def update_batch_size(self):
        batch_size = self.previous.out_shape[0]
        
        self.out_shape = (
            batch_size,
            *self.out_shape[1:]
        )

        self.xp4D.update(batch_size)

    # 3D tensor --> 3D tensor
    def forward_pass(self):
        self.a, self.z = self.xp4D.max(self.previous.a, self.previous.z)
        return self.a

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        return self.xp4D.distribute(dz_next)

class MinPool(MaxPool):
    # 3D tensor --> 3D tensor
    def forward_pass(self):
        self.a, self.z = self.xp4D.min(self.previous.a, self.previous.z)
        return self.a

# 4D tensor --> 4D tensor
class AveragePool(MaxPool):
    # 3D tensor --> 3D tensor
    def forward_pass(self):
        self.a, self.z = self.xp4D.mean(self.previous.a, self.previous.z)
        return self.a

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        return self.xp4D.scaleup(dz_next)
    
# 4D tensor --> 4D tensor (the last dimension is single-elmenet array)
class GlobalMaxPool(MaxPool):
    def __init__(self):
        self.size = None
        self.previous = None
        self.out_shape = None

        # activations
        self.df = None # copied from the previous layer
        
        # parameters
        self.a = None
        self.z = None

        # wrapper
        self.xp4D = None

    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        self.size = previous.out_shape[2:]
        
        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            1, 1
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, 1)

# 4D tensor --> 4D tensor (the last dimension is single-elmenet array)
class GlobalMinPool(MinPool):
    # 3D tensor --> 3D tensor
    def __init__(self):
        self.size = None
        self.previous = None
        self.out_shape = None

        # activations
        self.df = None # copied from the previous layer
        
        # parameters
        self.a = None
        self.z = None

        # wrapper
        self.xp4D = None

    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        self.size = previous.out_shape[2:]
        
        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            1, 1
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, 1)

# 4D tensor --> 4D tensor (the last dimension is single-elmenet array)
class GlobalAveragePool(AveragePool):
    def __init__(self):
        self.size = None
        self.previous = None
        self.out_shape = None

        # activations
        self.df = None # copied from the previous layer
        
        # parameters
        self.a = None
        self.z = None

        # wrapper
        self.xp4D = None

    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        self.size = previous.out_shape[2:]
        
        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            1, 1
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, 1)

# 4D tensor --> 4D tensor
class AdaptiveMaxPool(MaxPool):
    def __init__(self, out_size=(1, 1), stride=1):
        self.out_size = out_size
        super().__init__(stride=stride)

    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        
        self.size = (
            previous.out_shape[2] - (self.out_size[0] - 1) * self.stride, 
            previous.out_shape[3] - (self.out_size[1] - 1) * self.stride, 
        )

        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            *self.out_size,
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, self.stride)

# 4D tensor --> 4D tensor
class AdaptiveMinPool(MinPool):
    def __init__(self, out_size=(1, 1), stride=1):
        self.out_size = out_size
        super().__init__(stride=stride)

    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        
        self.size = (
            previous.out_shape[2] - (self.out_size[0] - 1) * self.stride, 
            previous.out_shape[3] - (self.out_size[1] - 1) * self.stride, 
        )

        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            *self.out_size,
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, self.stride)

# 4D tensor --> 4D tensor
class AdaptiveAveragePool(AveragePool):
    def __init__(self, out_size=(1, 1), stride=1):
        self.out_size = out_size
        super().__init__(stride=stride)

    def fuse(self, previous):
        self.previous = previous
        self.df = previous.df
        
        self.size = (
            previous.out_shape[2] - (self.out_size[0] - 1) * self.stride, 
            previous.out_shape[3] - (self.out_size[1] - 1) * self.stride, 
        )

        self.out_shape = (
            previous.out_shape[0],
            previous.out_shape[1],
            *self.out_size,
        )

        self.xp4D = cmethods.I2x_pool4D(*previous.out_shape, *self.size, self.stride)