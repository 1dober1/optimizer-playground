class GD:
    def __init__(self, lr=None):
        self.lr = lr

    def step(self, w, grad):
        return w - self.lr * grad
