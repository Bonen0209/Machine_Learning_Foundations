import argparse
import random
import numpy as np
from helper import load_data

class Perceptron_Learning_Algorithm(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dim = X.shape[1]
        self.n = X.shape[0]

    def sign(self, input):
        if input > 0:
            return 1
        else:
            return -1

    def run(self, rand=False, alpha=1):
        update = 0
        w = np.zeros(self.dim)

        cands = list(range(self.n))
        if rand:
            cands = random.sample(cands, self.n)
        
        count = 0
        done = True
        while True:
            idx = cands[count]

            if self.sign(np.dot(w, self.X[idx])) != self.Y[idx]:
                w += alpha * self.Ｙ[idx] * self.Ｘ[idx]
                update += 1
                done = False

            count += 1

            if count == self.n:
                if done:
                    break
                else:
                    count = 0
                    done = True
        
        print('Done PLA')

        return update

    def run_train(self, pocket=True, rand=True, alpha=1, update_times=100):
        w = np.zeros(self.dim)
        wp = np.zeros(self.dim)

        cands = list(range(self.n))
        if rand:
            cands = random.sample(cands, self.n)

        while update_times:
            mistakes = []

            count = 0
            while True:
                idx = cands[count]

                if self.sign(np.dot(w, self.X[idx])) != self.Y[idx]:
                    mistakes.append(idx)

                count += 1

                if count == self.n:
                    break
            
            mistake = random.choice(mistakes)

            w += alpha * self.Ｙ[mistake] * self.Ｘ[mistake]

            wp_error = 0
            w_error = 0

            count = 0
            while True:
                idx = count

                if self.sign(np.dot(wp, self.X[idx])) != self.Y[idx]:
                    wp_error += 1

                if self.sign(np.dot(w, self.X[idx])) != self.Y[idx]:
                    w_error += 1

                count += 1

                if count == self.n:
                    break
            
            if w_error < wp_error:
                wp = np.copy(w)

            update_times -= 1
        
        print('Done Pocket')
        
        if pocket:
            return wp
        else:
            return w

    def run_test(self, w, X_test, Y_test):
        test_n = X_test.shape[0]

        w_error = 0
        count = 0
        while True:
            idx = count

            if self.sign(np.dot(w, X_test[idx])) != Y_test[idx]:
                w_error += 1

            count += 1

            if count == test_n:
                break

        return w_error

def main():
    # # Problem 6
    # parser = argparse.ArgumentParser(description='Perceptron Learning Algorithm')
    # parser.add_argument('--input_file', default='./HW1/hw1_6_train.dat', help='path of input data')
    
    # args = parser.parse_args()

    # X, Y = load_data(args.input_file)

    # PLA = Perceptron_Learning_Algorithm(X, Y)
    # print(PLA.run(rand=False))

    # # Problem 7
    # parser = argparse.ArgumentParser(description='Perceptron Learning Algorithm')
    # parser.add_argument('--train_file', default='./HW1/hw1_7_train.dat', help='path of train data')
    # parser.add_argument('--test_file', default='./HW1/hw1_7_test.dat', help='path of test data')
    # parser.add_argument('--repeat_times', default=1126, help='repeat times')
    
    # args = parser.parse_args()

    # X_train, Y_train = load_data(args.train_file)
    # X_test, Y_test = load_data(args.test_file)

    # PLA = Perceptron_Learning_Algorithm(X_train, Y_train)

    # print(PLA.run_test(PLA.run_train(), X_test, Y_test))

if __name__ == '__main__':
    main()
