import numpy as np
import matplotlib.pyplot as plt

class PNRays(object):
    def __init__(self, size):
        self.size = size

        self.X, self.Y, self.Theta = self._getdata(self.size)
    
    def reset_data(self):
        np.random.seed()
        self.X, self.Y, self.Theta = self._getdata(self.size)

    def _getdata(self, size):
        X = np.random.uniform(low=-1.0, high=1.0, size=size)
        X = np.sort(X)

        Y = np.where(X >= 0, 1.0, -1.0)
        noise = np.random.choice([1.0, -1.0], p=[0.8, 0.2], size=size)
        Y = Y * noise

        Theta = []
        Theta.append(np.median([-1.0, X[0]]))
        for idx in range(size - 1):
            Theta.append(np.median(X[idx:idx+2]))
        Theta.append(np.median([X[-1], 1.0]))

        return X, Y, Theta
    
    def _h(self, x, theta, s):
        return s * np.where((x - theta) >= 0, 1.0, -1.0)

    def _Eout(self, theta, s):
        return 0.5 + 0.3 * s * (np.abs(theta) - 1)

    def _Ein(self, theta, s):
        Ein = 0
        for x, y in zip(self.X, self.Y):
            if self._h(x, theta, s) != y:
                Ein += 1
        
        return Ein / self.size
    
    def decision_stump(self, multiprocessing=False):
        best_theta, best_s = 0.0, 0
        min_Ein = np.inf

        if multiprocessing:
            self.reset_data()

        for theta in self.Theta:
            for s in [-1, 1]:
                tmp_Ein = self._Ein(theta, s)
                if tmp_Ein <= min_Ein:
                    min_Ein = tmp_Ein
                    best_theta, best_s = theta, s

        return best_theta, best_s, min_Ein, self._Eout(best_theta, best_s)

    @staticmethod
    def plot_hist(histogram, title, x_label, y_label, out_dir, bins=100):
        hist, bins = np.histogram(histogram, bins=bins)
        width = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.savefig(out_dir + title.split()[-1])