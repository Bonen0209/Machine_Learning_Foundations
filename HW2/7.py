import argparse
from pnrays import PNRays
from tqdm import tqdm

SIZE = 20
ITER = 1000

def main():
    parser = argparse.ArgumentParser(description='Problem 7')
    parser.add_argument('--output_dir', default='./HW2/', help='path of output directory')
    
    args = parser.parse_args()

    Ray = PNRays(SIZE)
    Eins= []
    Eouts = []
    Diffs = []

    for _ in tqdm(range(ITER)):
        _, _, Ein, Eout = Ray.decision_stump()
        Eins.append(Ein)
        Eouts.append(Eout)
        Diffs.append(Ein - Eout)

        Ray.reset_data()

    print(sum(Eins)/len(Eins))
    print(sum(Eouts)/len(Eouts))
    print(sum(Diffs)/len(Diffs))
    
    Ray.plot_hist(Diffs, 'Problem 7', 'Ein - Eout', 'frequency', args.output_dir)


if __name__ == "__main__":
    main()