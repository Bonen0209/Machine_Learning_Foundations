import argparse
from pnrays import PNRays
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

SIZE = 2000
ITER = 1000

def main():
    parser = argparse.ArgumentParser(description='Problem 8')
    parser.add_argument('--output_dir', default='./HW2/', help='path of output directory')
    
    args = parser.parse_args()

    Ray = PNRays(SIZE)
    Iters = [True] * ITER
    Eins = []
    Eouts = []
    Diffs = []

    with ProcessPoolExecutor() as executor:
        for _, _, Ein, Eout in tqdm(executor.map(Ray.decision_stump, Iters), total=ITER):
            Eins.append(Ein)
            Eouts.append(Eout)
            Diffs.append(Ein - Eout)

    print(sum(Eins)/len(Eins))
    print(sum(Eouts)/len(Eouts))
    print(sum(Diffs)/len(Diffs))
    
    Ray.plot_hist(Diffs, 'Problem 8', 'Ein - Eout', 'frequency', args.output_dir)


if __name__ == "__main__":
    main()