class GD:
    def __init__(self, lr=None):
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        self.lr = lr

    def step(self, w, grad):
        return w - self.lr * grad
