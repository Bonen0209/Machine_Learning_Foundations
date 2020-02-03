import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression(object):
    def __init__(self, train_path, test_path, etas, T):
        self.etas = etas
        self.T = T
        self.train_path = train_path
        self.test_path = test_path

        self.X_train, self.Y_train = self._load_data(self.train_path)
        self.X_test, self.Y_test = self._load_data(self.test_path)

        self.dim = self.X_train.shape[1]
        self.w = np.zeros((1, self.dim))

        self.Eins = {}
        self.Eouts = {}
        for eta in self.etas:
            for SGD in [True, False]:
                key = (eta, SGD)

                self.Eins[key] = []
                self.Eouts[key] = []


    def _load_data(self, filename):
        df = pd.read_csv(filename, sep=r'\s+')
        data = df.values    
        X = np.c_[np.ones((len(data), 1)), data[:, :-1]]
        Y = data[:, -1:]

        return X, Y

    def _init_w(self):
        self.w = np.zeros((1, self.dim))

    def fit(self, eta, SGD):
        key = (eta, SGD)

        for t in range(self.T):
            self.w -= eta * self._gradientEin(t, self.w, self.X_train, self.Y_train, SGD=SGD)

            self.Eins[key].append(self._Ein())
            self.Eouts[key].append(self._Eout())
        
        self._evaluate()
        self._init_w()

    def run(self):
        for SGD in [True, False]:
            for eta in self.etas:
                self.fit(eta=eta, SGD=SGD)    

    def _Ein(self):
        return np.sum(self._sign(np.dot(self.X_train, self.w.T)) != self.Y_train) / self.X_train.shape[0]

    def _Eout(self):
        return np.sum(self._sign(np.dot(self.X_test, self.w.T)) != self.Y_test) / self.X_test.shape[0]
    
    def _evaluate(self):
        print(f'Ein: {self._Ein()}')
        print(f'Eout: {self._Eout()}')
    
    def plot_Eout(self):
        for key in self.Eouts:
            plt.plot(range(self.T), self.Eouts[key], label=f'Eta: {key[0]}{", SGD" * key[1]}')

        plt.legend()
        plt.title('Problem 7')
        plt.xlabel('T')
        plt.ylabel('Eout')
        plt.savefig('./HW3/7.png')
        plt.close()

    def plot_Ein(self):
        for key in self.Eouts:
            plt.plot(range(self.T), self.Eins[key], label=f'Eta: {key[0]}{", SGD" * key[1]}')

        plt.legend()
        plt.title('Problem 8')
        plt.xlabel('T')
        plt.ylabel('Ein')
        plt.savefig('./HW3/8.png')
        plt.close()

    @staticmethod
    def _theta(s):
        return 1 / (1 + np.exp(-s))
    
    @staticmethod
    def _gradientEin_single(w, x, y):
        return Logistic_Regression._theta(-y * np.dot(x, w.T)) * (-y * x)
    
    @staticmethod
    def _gradientEin(t, w, X, Y, SGD):
        if SGD:
            index = t % X.shape[0]
            return Logistic_Regression._gradientEin_single(w, X[None, index], Y[None, index])
        else:
            return np.sum(Logistic_Regression._gradientEin_single(w, X, Y), axis=0).reshape(1, -1) / X.shape[0]

    @staticmethod
    def _sign(input):
        pos = input > 0
        neg = input <= 0

        return pos * 1 + neg * -1