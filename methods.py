import numpy as np

images = [0, 0, 0, 0, 0, 0]

# stuff
import matplotlib.pyplot as plt

def show_grid(images, info_show=False):
    fig, axes = plt.subplots(len(images) // 2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            if info_show:
                for k in range(images[i].shape[0]):
                    for j in range(images[i].shape[1]):
                        # Center the text on each pixel (j=x, i=y)
                        ax.text(j, k, str(images[i][k, j]), color='red', 
                                ha='center', va='center', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off') # Hide unused subplots
    plt.tight_layout()
    plt.show()

# math operations
def align_and_pad(x_shape, k_shape, stride):
    u, d, l, r = 0, 0, 0, 0 # up, down, left, right padding

    # check for x's boundaries compatibility with kernel's sliding 
    # and return the padding needed to maintain a perfect match 
    # between the x-kernel moving window and the kernel.

    r_row, r_col = (x_shape[0] - k_shape[0] + 1) % stride, (x_shape[1] - k_shape[1] + 1) % stride
    pad_room_row = stride - r_row 
    pad_room_col = stride - r_col 
    
    if r_row:
        pad = int(pad_room_row / 2)
        if pad_room_row % 2:
            u, d = pad + 1, pad
        else:
            u = d = pad
    if r_col:
        pad = int(pad_room_col / 2)
        if pad_room_col % 2:
            l, r = pad, pad + 1
        else:
            l = r = pad
    
    return (u, d), (l, r)

# initializers
def he_normal(n_in, n_out, rng=np.random):
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0.0, std, size=(n_out, n_in))

def he_uniform(n_in, n_out, rng=np.random):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_out, n_in))

def glorot_normal(n_in, n_out, rng=np.random):
    std = np.sqrt(2.0 / (n_in + n_out))
    return rng.normal(0.0, std, size=(n_out, n_in))

def glorot_uniform(n_in, n_out, rng=np.random):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_out, n_in))

def zeros(shape):
    return np.zeros(shape)

# activations
def sigmoid(vec):
   return 1 / (1 + np.exp(-vec))

def sigmoid_derivative(vec):
    a = sigmoid(vec)
    return a * (1 - a)

def softmax(vec):
    vec_exp = np.exp(vec - np.max(vec, axis=1, keepdims=True))
    return vec_exp / np.sum(vec_exp, axis=1, keepdims=True)

def softmax_derivative(vec):
    S = softmax(vec)
    dS = np.outer(-S, S) # issue
    diag_indices = np.diag_indices(dS.shape[1])
    dS[:, *diag_indices] += S
    return dS

def ReLU(vec):
    if isinstance(vec[0], (np.number, float, int)):
        return np.maximum(0, vec)
    return np.array([ReLU(r) for r in vec], dtype=float)

def Leaky_ReLU(vec):
    if isinstance(vec[0], (np.number, float, int)):
        return np.maximum(0.1 * vec, vec)
    return np.array([Leaky_ReLU(r) for r in vec])

# Loss functions
def MSE(o, y):
    erorr = o - y
    return np.sum(erorr * erorr, axis=1) / y.shape[0]

def BCE(o, y):
    return -np.mean(y * np.log(o + 1e-8) + (1 - y) * np.log(1 - o + 1e-8), axis=1)

def CCE(o, y):
    return -np.sum(y * np.log(o + 1e-9), axis=1)

# optimizers
class Momentem():
    def __init__(self, beta=0.9, lr=0.001):
        self.NL = 1
        self.lr = lr
        self.β = beta

    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL

    def func(self, dW, dB, l):
        self.m_w[l] = self.β * self.m_w[l] + (1 - self.β) * dW
        self.m_b[l] = self.β * self.m_b[l] + (1 - self.β) * dB

        return self.m_w[l] * self.lr, self.m_b[l] * self.lr 

class Nestrov_A():
    def __init__(self, beta=0.9, lr=0.001):
        self.NL = 1
        self.lr = lr
        self.β = beta
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL

    def func(self, dW, dB, l):
        # apply acceleration
        self.m_w[l] = self.β * self.m_w[l] + self.lr * dW
        dW = self.β * self.m_w[l] + self.lr * dW
        
        self.m_b[l] = self.β * self.m_b[l] + self.lr * dB
        dB = self.β * self.m_b[l] + self.lr * dB

        return dW, dB 

class AdaGrad():
    def __init__(self, lr=0.001):
        self.NL = 1
        self.lr = lr
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.v_w = [0] * self.NL
        self.v_b = [0] * self.NL

    def func(self, dW, dB, l):
        # dw
        self.v_w[l] = self.v_w[l] + dW * dW
        dW = self.lr * dW / (np.sqrt(self.v_w[l]) + 1e-8) 

        # db
        self.v_b[l] = self.v_b[l] + dB * dB
        dB = self.lr * dB / (np.sqrt(self.v_b[l]) + 1e-8) 

        return dW, dB

class AdaDelta():
    def __init__(self, beta=0.9, lr=0.001):
        self.NL = 1
        self.lr = lr
        self.β = beta
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL

    def func(self, dW, dB, l):
        # dw
        self.m_w[l] = self.β * self.m_w[l] + (1 - self.β) * dW * dW
        RMS_1 = np.sqrt(self.m_w[l] + 1e-8)
        delta_w = self.lr / RMS_1 * dW

        RMS_2 = np.sqrt(self.v_w[l] + 1e-8)
        self.v_w[l] = self.β * self.v_w[l] + (1 - self.β) * delta_w * delta_w
        
        dW = RMS_2 / RMS_1 * dW
        
        # db 
        self.m_b[l] = self.β * self.m_b[l] + (1 - self.β) * dB * dB
        RMS_1 = np.sqrt(self.m_b[l] + 1e-8)
        delta_b = self.lr / RMS_1 * dB

        RMS_2 = np.sqrt(self.v_b[l] + 1e-8)
        self.v_b[l] = self.β * self.v_b[l] + (1 - self.β) * delta_b * delta_b
        
        dB = RMS_2 / RMS_1 * dB

        return dW, dB

class RMSProp():
    def __init__(self, beta=0.9, lr=0.001):
        self.NL = 1
        self.lr = lr
        self.β = beta
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.v_w = [0] * self.NL
        self.v_b = [0] * self.NL

    def func(self, dW, dB, l):
        # dw
        self.v_w[l] = self.β * self.v_w[l] + (1 - self.β) * dW * dW
        dW = self.lr * dW / np.sqrt(self.v_w[l] + 1e-8)
        
        # db
        self.v_b[l] = self.β * self.v_b[l] + (1 - self.β) * dB * dB
        dB = self.lr * dB / np.sqrt(self.v_b[l] + 1e-8)

        return dW, dB

class AdaMax():
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        self.NL = 1
        self.t, self.lr = 0, lr
        self.β1, self.β2 = beta1, beta2
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL

    def func(self, dW, dB, l):
        # increment the time
        if l == self.N - 1: self.t += 1

        # dw
        self.m_w[l] = (self.β1 * self.m_w[l] + (1 - self.β1) * dW)
        self.v_w[l] = (self.β2 * self.v_w[l] + (1 - self.β2) * dW * dW)

        m_hat = self.m_w[l] / (1 - np.pow(self.β1, self.t))
        u = [max(vw, dw) for vw, dw in zip(self.β2 * self.v_w[l], np.abs(dW))]
        u = np.array(u, dtype=float)
        
        dW = self.lr / u * m_hat
        
        # db
        self.m_b[l] = (self.β1 * self.m_b[l] + (1 - self.β1) * dB)
        self.v_b[l] = (self.β2 * self.v_b[l] + (1 - self.β2) * dB * dB)

        m_hat = self.m_b[l] / (1 - np.pow(self.β1, self.t))
        u = [max(vb, db) for vb, db in zip(self.β2 * self.v_b[l], np.abs(dB))]
        u = np.array(u, dtype=float)

        dB = self.lr / u * m_hat

        return dW, dB

class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        self.NL = 1
        self.t, self.lr = 0, lr
        self.β1, self.β2 = beta1, beta2
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL
    
    def func(self, dW, dB, l):
        # increment the time
        if l == self.N - 1: self.t += 1

        # dw
        self.m_w[l] = (self.β1 * self.m_w[l] + (1 - self.β1) * dW)
        self.v_w[l] = (self.β2 * self.v_w[l] + (1 - self.β2) * dW * dW)

        m_hat = self.m_w[l] / (1 - np.pow(self.β1, self.t)) 
        v_hat = self.v_w[l] / (1 - np.pow(self.β2, self.t))

        dW = self.lr * m_hat / (1e-8 + np.sqrt(v_hat))
        # db
        self.m_b[l] = (self.β1 * self.m_b[l] + (1 - self.β1) * dB)
        self.v_b[l] = (self.β2 * self.v_b[l] + (1 - self.β2) * dB * dB)

        m_hat = self.m_b[l] / (1 - np.pow(self.β1, self.t))
        v_hat = self.v_b[l] / (1 - np.pow(self.β2, self.t))
        
        dB = self.lr * m_hat / (1e-8 + np.sqrt(v_hat))

        return dW, dB

class nAdam():
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        self.NL = 1 
        self.t, self.lr = 0, lr
        self.β1, self.β2 = beta1, beta2
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL

    def func(self, dW, dB, l):
        # increment the time
        if l == self.N - 1: self.t += 1

        # dw
        self.m_w[l] = self.β1 * self.m_w[l] + (1 - self.β1) * dW
        self.v_w[l] = self.β2 * self.v_w[l] + (1 - self.β2) * dW * dW

        m_hat = self.m_w[l] / (1 - np.pow(self.β1, self.t)) 
        v_hat = self.v_w[l] / (1 - np.pow(self.β2, self.t))
        
        crer = self.β1 * m_hat + (1 - self.β1) / (1 - np.pow(self.β1, self.t))  * dW
        dW = self.lr / (1e-8 + np.sqrt(v_hat)) * crer
        
        # db
        self.m_b[l] = self.β1 * self.m_b[l] + (1 - self.β1) * dB
        self.v_b[l] = self.β2 * self.v_b[l] + (1 - self.β2) * dB * dB

        m_hat = self.m_b[l] / (1 - np.pow(self.β1, self.t))
        v_hat = self.v_b[l] / (1 - np.pow(self.β2, self.t))
        
        crer = self.β1 * m_hat + (1 - self.β1) / (1 - np.pow(self.β1, self.t))  * dB
        dB = self.lr / (1e-8 + np.sqrt(v_hat)) * crer
        
        return dW, dB

class AMSGrad():
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001):
        self.NL = 1
        self.lr = lr
        self.β1, self.β2 = beta1, beta2
    
    @property
    def N(self):
        return self.NL
    
    @N.setter
    def N(self, val):
        self.NL = val
        self.m_w, self.v_w = [0] * self.NL, [0] * self.NL
        self.m_b, self.v_b = [0] * self.NL, [0] * self.NL

    def func(self, dW, dB, l):
        # dw
        prev_v_w = self.v_w[l]
        self.m_w[l] = (self.β1 * self.m_w[l] + (1 - self.β1) * dW)
        self.v_w[l] = (self.β2 * self.v_w[l] + (1 - self.β2) * dW * dW)

        v_hat = [max(vw, pvw) for vw, pvw in zip(self.v_w[l], prev_v_w)]
        v_hat = np.array(v_hat, dtype=float)
        
        dW = self.lr / (1e-8 + np.sqrt(v_hat)) * self.m_w[l]
        
        # db
        prev_v_b = self.v_b[l]
        self.m_b[l] = (self.β1 * self.m_b[l] + (1 - self.β1) * dB)
        self.v_b[l] = (self.β2 * self.v_b[l] + (1 - self.β2) * dB * dB)

        v_hat = [max(vb, pvb) for vb, pvb in zip(self.v_b[l], prev_v_b)]
        v_hat = np.array(v_hat, dtype=float)
        
        dB = self.lr / (1e-8 + np.sqrt(v_hat)) * self.m_b[l]
        return dW, dB


derivatives = {
            # activation function : df/dx (for any x)
            sigmoid: sigmoid_derivative,
            softmax: softmax_derivative,
            ReLU: lambda v : (v > 0),
            Leaky_ReLU: lambda v : np.where(v > 0, 1, 0.1),
            # loss function : dl/dz (for z the last layer's weighted sum)
            MSE: lambda l,y: 2 * (l.a - y) * l.df(l.z),
            BCE: lambda l,y: l.a - y,
            CCE: lambda l,y: l.a - y
            }
