import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from helper import load_data, plot_hist
from perceptron_learning_algorithm import Perceptron_Learning_Algorithm

def main():
    parser = argparse.ArgumentParser(description='Problem 8')
    parser.add_argument('--train_file', default='./HW1/hw1_7_train.dat', help='path of train data')
    parser.add_argument('--test_file', default='./HW1/hw1_7_test.dat', help='path of test data')
    parser.add_argument('--repeat_times', default=1126, help='repeat times')
    
    args = parser.parse_args()

    X_train, Y_train = load_data(args.train_file)
    X_test, Y_test = load_data(args.test_file)

    PLA = Perceptron_Learning_Algorithm(X_train, Y_train)

    total = 0
    ws = []
    errors = []

    with ProcessPoolExecutor() as executor:
        results = [executor.submit(PLA.run_train, False) for _ in range(args.repeat_times)]

        for f in as_completed(results):
            result = f.result()
            ws.append(result)
    
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(PLA.run_test, w, X_test, Y_test) for w in ws]

        for f in as_completed(results):
            result = f.result()
            errors.append(result)

    total = sum(errors)
    avg = total / (args.repeat_times * X_test.shape[1])
    print(avg)

    errors = np.asarray(errors) / X_test.shape[0]
    plot_hist(errors, 'Problem 8', 'error rate', 'frequency', 200)
    
if __name__ == '__main__':
    main()