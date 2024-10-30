# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_stdout(stdout_file):
    train_inner_log, train_log, eval_log = [], [], []
    with open(stdout_file) as f:
        for line in f.readlines():
            if "[train_inner][INFO]" in line:
                train_inner_log.append(eval(line.split(" - ")[-1][:-1]))
            if "[train][INFO]" in line:
                train_log.append(eval(line.split(" - ")[-1][:-1]))
            if "[eval][INFO]" in line:
                eval_log.append(eval(line.split(" - ")[-1][:-1]))
    dfti = pd.DataFrame(train_inner_log)
    dft = pd.DataFrame(train_log)
    dfe = pd.DataFrame(eval_log)
    return dfti, dft, dfe


def plot_learning_curve(dfti, filename):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        dfti["num_updates"].astype("float"),
        dfti["loss"].astype("float"),
        "m-",
        label="loss",
    )
    ax2.plot(
        dfti["num_updates"].astype("float"),
        dfti["lr_default"].astype("float"),
        "k-",
        label="lr",
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("loss", color="k")
    ax2.set_ylabel("lr", color="k")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85))
    plt.title("Learning curve")
    fig.savefig(filename)


def plot_val_loss_acc(dfe, filename):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        dfe["eval_num_updates"].astype("float"),
        dfe["eval_loss"].astype("float"),
        "m-",
        label="val_loss",
    )
    ax2.plot(
        dfe["eval_num_updates"].astype("float"),
        dfe["eval_accuracy"].astype("float"),
        "k-",
        label="val_acc",
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("loss", color="k")
    ax2.set_ylabel("accuracy", color="k")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85))
    plt.title("Validation results")
    fig.savefig(filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    # output_dir = "stdout/eat_base/hypertuning/esc50"
    # fold = 0
    # for fold in range(5):
    stdout_file = os.path.join(args.output_dir, f"fold_{args.fold}/stdout.log")
    dfti, dft, dfe = load_stdout(stdout_file)
    plot_learning_curve(
        dfti,
        filename=os.path.join(
            args.output_dir, f"fold_{args.fold}/learning_curve.png"
        ),
    )
    plot_val_loss_acc(
        dfe,
        filename=os.path.join(
            args.output_dir, f"fold_{args.fold}/val_loss_acc.png"
        ),
    )
