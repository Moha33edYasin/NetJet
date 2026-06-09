from Layers import *
from methods import SGD

# for plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Network
class nn():
    def __init__(self, *sequence):
        self.possible_outcomes = None
        self.optim = None
        self.c = None 
        self.dc = None
        self.lr = None

        self.input_shape = None
        self.starter = None
        self.stages = sequence
        self.layers = [s for s in sequence if isinstance(s, (Dense, Conv))]

    def compile(self, input_shape, possible_outcomes=None, cost=None, dcost=None, lr=0.01, optimizer=None):
        
        ''' connect this network '''
        t = perf_counter()
        # Input(...) : works like starting point for the nn        
        self.starter = Input(input_shape)
        
        self.input_shape = input_shape
        self.stages[0].fuse(self.starter)
        
        completed = int(1 / len(self.stages) * 100)        
        print(f"({completed:.0f}%) : fused {type(self.stages[0]).__name__} (0) --> Input Node.")
        
        for i, s in enumerate(self.stages[1:]):
            s.fuse(self.stages[i])
            completed = int((i + 2) / len(self.stages) * 100)
            print(f"({completed:.0f}%) : fused {type(s).__name__} ({i + 1}) --> {type(self.stages[i]).__name__} ({i}).")

        self.input_shape = input_shape
        self.possible_outcomes = np.array(possible_outcomes)
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # inform the optimizer of the NN's structure by
        # passing the number of stages to the optimizer
        if optimizer != None:
            optimizer.N = len(self.layers)
        else:
            self.optim = SGD(lr)
        print(f"{type(self.optim).__name__} optimization is configured.")

        print(f"(*) The network is compiled successfully.       ({perf_counter() - t :.2f}s)\n")

    def resize_batch(self, batch_size):
        self.input_shape = (
                batch_size,
                self.input_shape[1],
                self.input_shape[2],
                self.input_shape[3]
        )        

        self.starter.update_batch_size(batch_size)
        [s.update_batch_size() for s in self.stages]

    def feedforward(self, a):
        self.starter.activate(a)
        return [s.forward_pass() for s in self.stages][-1]

    def loss(self, y):
        return self.c(self.output_vector(), y).sum() / y.shape[0]

    def backprop(self, y):

        ''' we will take the derivitative of the cost associated with 
            a particular datapoint with respect to each neuron in 
            the last layer. '''
        
        # initialize dz from the last layer
        if self.dc == None:
            try:
                dz = derivatives[self.c](self.stages[-1], y)
            except:
                raise ValueError(f"No differential expression is assigned to ({self.c.__name__})")
        else:
            dz = self.dc(self.stages[-1], y) * self.df(self.stages[-1].z)
        
        i = len(self.layers) - 1
        # now, we will work things backward
        for s in reversed(self.stages):
            if s.previous:
                if s.previous.previous and s.previous.z is not None:
                    dz_next = s.backward_pass(dz)

            if s in self.layers:
                dw, db = s.calculate_gradient(dz)

                # update gradient
                dw, db = self.optim.func(dw, db, i)
                s.w -= dw
                s.b -= db
                i -= 1

            dz = dz_next

    def learn(self, x_data, y_data, targets, epochs=1, Debug_plot=False):
        acc_traj_e = []
        loss_traj_e = []

        data = np.array(list(zip(x_data, y_data, targets)), dtype=tuple)
        batch_size = self.input_shape[0]
        
        if Debug_plot:
            loss_traj_b, acc_traj_b, n_batch, n_epoch = [], [], [], []
            loss_per_epoch, n_correct_per_epoch, n = 0, 0, 1
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            (acc_line_b,) = ax1.plot([], [], "-b", label="Accuracy per Batch")
            (loss_line_b,) = ax1.plot([], [], "--y", label="Loss per Batch")
            (acc_line_e,) = ax2.plot([], [], "o-r", label="Accuracy per Epoch")

            fig.autofmt_xdate()

            def frames_generator():
                for epoch in range(epochs):
                    for i in range(0, data.shape[0], batch_size):
                        yield epoch, i

            def init():
                ax1.set_xlim(1, data.shape[0] // batch_size + 2)
                ax1.set_ylim(0, 105)
                ax2.set_xlim(1, epochs + 0.5)
                ax2.set_ylim(0, 105)
                return (acc_line_b, loss_line_b, acc_line_e)

            def update(frame):
                nonlocal loss_per_epoch, n_correct_per_epoch, n
                
                epoch, i = frame
                if i == 0:
                    acc_traj_e.append(round(n_correct_per_epoch / data.shape[0] * 100, 2))
                    loss_traj_e.append(loss_per_epoch)
                    n_epoch.append(epoch + 1)
                    
                    loss_traj_b.clear()
                    acc_traj_b.clear()
                    n_batch.clear()
                    n_correct_per_epoch = 0
                    n = 1

                batch = data[i : batch_size + i]
            
                # shuffle the batch
                np.random.default_rng().shuffle(batch)
                x, y, t = batch[:, 0], batch[:, 1], batch[:, 2]
                
                x = np.stack(x)
                y = np.stack(y)
                
                if x.shape[0] != self.input_shape[0]:
                    self.resize_batch(x.shape[0])
                
                self.feedforward(x)
                loss_per_batch = self.loss(y)
                self.backprop(y)
                
                n_correct_per_batch = np.count_nonzero(self.output() == t)
                acc_per_batch = np.round(n_correct_per_batch / x.shape[0] * 100, 2)

                # calculate loss and accuracy per epoch
                loss_per_epoch += loss_per_batch
                n_correct_per_epoch += n_correct_per_batch

                loss_traj_b.append(loss_per_batch)
                acc_traj_b.append(acc_per_batch)
                n_batch.append(n)

                acc_line_b.set_data(n_batch, acc_traj_b)
                loss_line_b.set_data(n_batch, loss_traj_b)
                acc_line_e.set_data(n_epoch, acc_traj_e)
                n += 1

                return (acc_line_b, loss_line_b, acc_line_e)

            start_time = perf_counter()
            _ = FuncAnimation(
                fig, 
                update, 
                init_func=init, 
                frames=frames_generator, 
                interval=1,
                cache_frame_data=False, 
                repeat=False
            )

            ax1.set_title("Training per Batch")
            ax2.set_title("Training per Epoch")
            ax1.set_ylabel("%")
            ax2.set_ylabel("%")
            ax1.legend(loc="best")
            ax2.legend(loc="best")
            ax1.grid(True)
            ax2.grid(True)
            plt.show()

        else:
            start_time = perf_counter()
            for epoch in range(epochs):
                # backpropagation
                loss, n_correct = 0, 0

                T = perf_counter()
                
                for i in range(0, data.shape[0], batch_size):
                    batch = data[i : batch_size + i]
            
                    # shuffle the batch
                    np.random.default_rng().shuffle(batch)
                    x, y, t = batch[:, 0], batch[:, 1], batch[:, 2]
                    
                    x = np.stack(x)
                    y = np.stack(y)
                    
                    if x.shape[0] != self.input_shape[0]:
                        self.resize_batch(x.shape[0])
                    
                    self.feedforward(x)                
                    loss += self.loss(y)
                    self.backprop(y)
                    
                    n_correct += np.count_nonzero(self.output() == t)

                # calculate loss and accuracy per epoch
                acc_percentage = n_correct / data.shape[0] * 100
                loss_traj_e.append(loss)
                acc_traj_e.append(round(acc_percentage, 2))
                print(f"(#) accuracy per epoch ({epoch + 1}): {acc_percentage:.2f}%       ({perf_counter() - T :.2f}s)")

        print(f"(*) Training is complete.       ({perf_counter() - start_time :.2f}s)\n")      
        return loss_traj_e, acc_traj_e

    def test(self, x_data, y_data, targets):
        loss_traj, accuarcy = [], []
        batch_size = self.input_shape[0]

        start_time = perf_counter()
        n, n_correct = 0, 0
        for i in range(0, x_data.shape[0], batch_size):
            x = x_data[i : batch_size + i]
            y = y_data[i : batch_size + i]
            t = targets[i : batch_size + i]
            
            if x.shape[0] != self.input_shape[0]:
                self.resize_batch(x.shape[0])

            self.feedforward(x)
            loss = self.loss(y)
            n_correct_per_batch = np.count_nonzero(self.output() == t)
            n_correct += n_correct_per_batch
        
            # calculate accuarcy per batch
            acc_percentage = np.round(n_correct_per_batch / x.shape[0] * 100, 2)

            loss_traj.append(loss)
            accuarcy.append(acc_percentage)

            print(f"(~) accuracy per batch ({n}):", f"{acc_percentage :.2f}%")
            n += 1

        print(f"($) Overall Accuracy:", f"{n_correct / x_data.shape[0] * 100 :.2f}%")
        print(f"(*) Testing is complete.       ({perf_counter() - start_time :.2f}s)\n")
        return loss_traj, accuarcy

    def output_vector(self):
        return self.stages[-1].a
    
    def output(self):
        last_stage = self.stages[-1].a
        out_index = last_stage.argmax(axis=1)
        return self.possible_outcomes[out_index]
  
# TODO: add branching feature
