# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as skm


def roc_auc(y, y_p, filename=None, plot=True):
    auc = skm.roc_auc_score(y, y_p)
    plt.figure()
    skm.RocCurveDisplay.from_predictions(y, y_p)
    plt.title("ROC curve, auc={:.04f}".format(auc))
    if filename is not None:
        plt.savefig(filename)
    return auc


def best_throshold(y, y_p, metric):
    df = {
        "th": [],
        "acc": [],
        "f1": [],
        "recall": [],
        "precision": [],
    }
    for th in np.arange(0, 1, 0.01):
        acc = skm.accuracy_score(y, (y_p >= th).astype("int"))
        f1 = skm.f1_score(y, (y_p >= th).astype("int"))
        recall = skm.recall_score(y, (y_p >= th).astype("int"))
        precision = skm.precision_score(y, (y_p >= th).astype("int"))
        df["th"].append(th)
        df["acc"].append(acc)
        df["f1"].append(f1)
        df["recall"].append(recall)
        df["precision"].append(precision)
    df = pd.DataFrame(df)
    best_th = df.th.iloc[df[metric].argmax()]
    print(
        "Best {}={:.04f} with threshold={}".format(
            metric, df[metric].max(), best_th
        )
    )
    return best_th


def binary_confusion_matrix(y, y_p, th, plot=False, filename=None):
    acc = skm.accuracy_score(y, (y_p >= th).astype("int"))
    f1 = skm.f1_score(y, (y_p >= th).astype("int"))
    recall = skm.recall_score(y, (y_p >= th).astype("int"))
    precision = skm.precision_score(y, (y_p >= th).astype("int"))
    cm = skm.confusion_matrix(y, (y_p >= th).astype("int"))
    results = {
        "cm": cm,
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
    }
    if plot:
        df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
        plt.figure()
        sns.heatmap(
            df_cm.astype("int"),
            annot=True,
            fmt="d",
            cbar=False,
            annot_kws={"fontsize": 16},
        )
        plt.title(
            "threshold={:.04f}, acc={:.02f}, f1={:.02f},\n recall={:.02f}, precision={:.02f}".format(
                th, acc, f1, recall, precision
            )
        )
        if filename is not None:
            plt.savefig(filename)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    df_ls = []
    for fold in range(5):
        df_ls.append(
            pd.read_csv(
                os.path.join(
                    args.output_dir, f"fold_{fold}", "test_results.csv"
                ),
                index_col=[0],
            )
        )
    df = pd.concat(df_ls)
    df["original_file_id"] = df["audio_file"].apply(
        lambda x: "_".join(x.split("/")[-1].split("_")[:-1])
    )
    df = (
        df.groupby(["original_file_id", "label"])["predict"]
        .max()
        .reset_index()
    )
    y_test = df["label"].to_numpy()
    y_p_test = df["predict"].to_numpy()
    auc_test = roc_auc(
        y_test,
        y_p_test,
        filename=os.path.join(args.output_dir, "roc_test.png"),
    )
    best_th_test = best_throshold(y_test, y_p_test, "f1")
    results_test = binary_confusion_matrix(
        y_test,
        y_p_test,
        best_th_test,
        plot=True,
        filename=os.path.join(args.output_dir, "cm_test.png"),
    )
    results = {
        "auc_test": auc_test,
        "f1_test": results_test["f1"],
        "cm_test": np.array2string(results_test["cm"], separator=","),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
