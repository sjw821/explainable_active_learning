import argparse
from al_loop import run_active_learning
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--method', choices=['al', 'al_ei'], required=True,
                        help="'al' for conventional AL, 'al_ei' for explainable intervention")
    parser.add_argument('--lambda_val', type=float, default=0.0,
                        help='Feature weight for noisy features when using al_ei')
    parser.add_argument('--rep', type=int, required=True)
    args = parser.parse_args()

    results = run_active_learning(
        dataname=args.dataname,
        method=args.method,
        rep=args.rep,
        lambda_val=args.lambda_val,
        
        n_iter=2
    )
    out = f"results_{args.dataname}_{args.method}_{args.lambda_val}_{args.rep}.pkl"
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {out}")