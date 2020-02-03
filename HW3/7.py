import argparse
from logistic_regression import Logistic_Regression

def main():
    parser = argparse.ArgumentParser(description='Problem 7')
    parser.add_argument('--train_file', default='./HW3/hw3_train.dat', help='path of train data')
    parser.add_argument('--test_file', default='./HW3/hw3_test.dat', help='path of test data')
    parser.add_argument('--etas', default=[0.01, 0.001], help='model parameters')
    parser.add_argument('--T', default=2000, help='model parameters')

    args = parser.parse_args()

    Log_R = Logistic_Regression(args.train_file, args.test_file, etas=args.etas, T=args.T)
    Log_R.run()

    Log_R.plot_Eout()

if __name__ == "__main__":
    main()