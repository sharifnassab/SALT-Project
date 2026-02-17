import numpy as np


class center_():
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.mu = 0
    def step(self, x):
        x_centered = x-self.mu
        x_tilde, mu, sigma = self.normalizer.step(x)
        self.mu = mu
        return x_centered, mu, sigma

class center_and_augment():  # centering + splitting
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.mu = None
    def step(self, x):
        if self.mu is None:
            self.mu = np.zeros_like(x)
        x_centered = x-self.mu
        x_concat = np.concatenate([x_centered, self.mu], axis=0)
        x_tilde, mu, sigma = self.normalizer.step(x)
        self.mu = mu
        return x_concat, mu, sigma