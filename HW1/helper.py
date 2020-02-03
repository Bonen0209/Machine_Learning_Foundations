import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def load_data(filename):
    df = pd.read_csv(filename, sep=r'\s+')
    data = df.values    
    X = np.c_[np.ones((len(data), 1)), data[:, :-1]]
    Y = data[:, -1:]

    return X, Y

def plot_hist(histogram, title, x_label, y_label, bins=100):
    hist, bins = np.histogram(histogram, bins=bins)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    plt.savefig('./HW1/' + title.split()[-1])

def main():
    parser = argparse.ArgumentParser(description='Datahelper')
    parser.add_argument('--input_file', default='./HW1/hw1_6_train.dat', help='path of input data')
    
    args = parser.parse_args()
    
    X, Y = load_data(args.input_file)
    print(X.shape)
    print(Y.shape)

if __name__ == '__main__':
    main()