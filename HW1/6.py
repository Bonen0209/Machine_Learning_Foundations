import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from helper import load_data, plot_hist
from perceptron_learning_algorithm import Perceptron_Learning_Algorithm

def main():
    parser = argparse.ArgumentParser(description='Problem 6')
    parser.add_argument('--input_file', default='./HW1/hw1_6_train.dat', help='path of input data')
    parser.add_argument('--repeat_times', default=1126, help='repeat times')
    
    args = parser.parse_args()

    X, Y = load_data(args.input_file)

    PLA = Perceptron_Learning_Algorithm(X, Y)

    total = 0
    updates = []

    with ProcessPoolExecutor() as executor:
        results = [executor.submit(PLA.run, True) for _ in range(args.repeat_times)]

        for f in as_completed(results):
            result = f.result()
            updates.append(result)

    total = sum(updates)
    avg = total / args.repeat_times
    print(avg)

    updates = np.asarray(updates)
    plot_hist(updates, 'Problem 6', 'updates', 'frequency')

if __name__ == '__main__':
    main()