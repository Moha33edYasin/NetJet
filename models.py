from Layers import *

# Network
class nn():
    def __init__(self, *sequence, possible_outcomes=None, cost=None, dcost=None, optimizer=None):
        self.possible_outcomes = np.array(possible_outcomes)
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # check for automated configuration
        self.stages = sequence
        for i, s in enumerate(sequence, 1):
            if isinstance(s, (Dense, Flatten)) and isinstance(sequence[i - 1], (Conv, MaxPool)):
                self.stages.insert(i + 1, Flatten())
            elif isinstance(s, (Conv, MaxPool)) and isinstance(sequence[i - 1], (Dense, Flatten)):
                self.stages.insert(i + 1, Reshape())
        self.layers = [s for s in sequence if isinstance(s, (Dense, Conv))]

        # inform the optimizer of the NN's structure by
        # passing the number of stages to the optimizer
        if optimizer != None:
            optimizer.N = len(self.stages)

    def feedforward(self, a):
        s = self.stages[0]

        if isinstance(s, Flatten):
            s.a = a = a.reshape(a.shape[0], -1)
            s.n = len(s.a[0])
        elif isinstance(s, Reshape):
            s.a = a = a.reshape(a.shape[0], *s.shape)
            s.n = len(s.a[0])
        else:
            # Input(a) works like starting point to the nn if a flatten 
            # or a reshape layer isn't that starting point 
            a = s.forward_pass(Input(a))
            
        for i, s in enumerate(self.stages[1:]):
            prev_s = self.stages[i]

            if isinstance(s, Reshape):
                if s.shape == -1:
                    next_stage = self.stages[i + 2]
                    n_branch = next_stage.branches
                    d1 = round(a.size / n_branch)
                    d2 = round(d1 / next_stage.shape[0])
                    s.shape = (d1, d2, next_stage.shape[0])
            a = s.forward_pass(prev_s)
        return a

    def loss(self, y):
        return self.c(self.output_vector(), y)

    def backprop(self, y):
        # we will take the derivitative of the cost associated with 
        # a particular datapoint with respect to each neuron in 
        # the last layer. 
        
        dW, dB = [], []
        
        # initialize dz from the last layer
        if self.dc == None:
            try:
                dz = derivatives[self.c](self.stages[-1], y)
            except:
                raise ValueError(f"No differential expression is assigned to ({self.c.__name__})")
        else:
            dz = self.dc(self.stages[-1], y) * self.df(self.stages[-1].z)

        # now, we will work things backward
        for i in reversed(range(len(self.stages))):
            s = self.stages[i]
            if s in self.layers:
                dw, db = s.calculate_gradient(dz)
                dw /= y.shape[0]
                db /= y.shape[0]
                if self.optim != None:
                    dw, db = self.optim.func(dw, db, i)
                
                dW.append(dw)
                dB.append(db)
            
            if s.previous:
                if s.previous.previous and len(s.previous.z):
                    dz = s.backward_pass(dz)

        dW.reverse()
        dB.reverse()
        return dW, dB 

    def learn(self, x_data, y_data, targets, lr=0.01, epochs=1, batch_size=1):
        loss_traj, accuracy = [], []

        data = np.array(list(zip(x_data, y_data, targets)), dtype=tuple)

        for epoch in range(epochs):
            # backpropagation
            loss, n_correct = 0, 0
            for i in range(0, data.shape[0], batch_size):
                batch = data[i : batch_size + i]
                
                # shuffle the batch
                np.random.default_rng().shuffle(batch)
                x, y, t = batch[:, 0], batch[:, 1], batch[:, 2]
                x = np.stack(x)
                y = np.stack(y)

                self.feedforward(x)
                loss += self.loss(y).sum() / y.shape[0]
                gW, gB = self.backprop(y)
                n_correct += np.count_nonzero(self.output() == t)
                
                # update gradient
                for i in range(len(self.layers)):
                    l = self.layers[i]
                    if self.optim == None:
                        # normal SGD
                        l.w -= gW[i] * lr
                        l.b -= gB[i] * lr
                    else:
                        l.w -= gW[i]
                        l.b -= gB[i]
            # calculate loss and accuracy per epoch
            try:
                acc_percentage = np.round(n_correct / data.shape[0] * 100, 2)
            except:
                raise ZeroDivisionError(f"batch_size shouldn't be zero.")

            loss_traj.append(loss)
            accuracy.append(acc_percentage)
            print(f"accuracy-per-epoch-{epoch + 1}:", f"{acc_percentage}%")

        return loss_traj, accuracy

    def test(self, x_data, y_data, targets, batch_size=1):
        loss_traj, accuarcy = [], []
        for i in range(0, x_data.shape[0], batch_size):
            x = x_data[i : batch_size + i]
            y = y_data[i : batch_size + i]
            t = targets[i : batch_size + i]

            self.feedforward(x)
            loss = self.loss(y).sum() / x.shape[0]
            n_correct = np.count_nonzero(self.output() == t)
        
            # calculate loss and accuarcy per batch
            try:
                acc_percentage = np.round(n_correct / x.shape[0] * 100, 2)
            except:
                raise ZeroDivisionError(f"(batch_size) should not be zero.")

            loss_traj.append(loss)
            accuarcy.append(acc_percentage)

            print(f"accuracy-per-batch-{(i // batch_size) + 1}:", f"{round(acc_percentage, 2)}%")
        print(f"overall_accuracy:", f"{round(sum(accuarcy) / ((i // batch_size) + 1), 2)}%")
        return loss_traj, accuarcy
        
    def set_optimizer(self, optimizer=None):
        self.optim = optimizer

    def output_vector(self):
        return self.stages[-1].a
    
    def output(self):
        last_stage = self.stages[-1].a
        out_index = last_stage.argmax(axis=1)
        return self.possible_outcomes[out_index]

  
# TODO: implement im2col + GEMM 
# ! implement fast convolution
# * add branching feature
