import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str)
    parser.add_argument("--output_data", type=str)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.raw_data}/X_train.csv")
    X_test  = pd.read_csv(f"{args.raw_data}/X_test.csv")
    y_train = pd.read_csv(f"{args.raw_data}/y_train.csv")
    y_test  = pd.read_csv(f"{args.raw_data}/y_test.csv")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Churn rate:    {y_train.mean().values[0]:.2%}")

    #Output folder
    os.makedirs(args.output_data, exist_ok=True)

    X_train.to_csv(f"{args.output_data}/X_train.csv", index=False)
    X_test.to_csv(f"{args.output_data}/X_test.csv",  index=False)
    y_train.to_csv(f"{args.output_data}/y_train.csv", index=False)
    y_test.to_csv(f"{args.output_data}/y_test.csv",  index=False)

if __name__ == "__main__":
    main()