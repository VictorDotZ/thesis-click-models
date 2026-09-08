from pathlib import Path
from typing import Callable
from argparse import ArgumentParser
from collections import defaultdict, Counter

import pandas as pd
import numpy as np


def create_search_record(session_id: str, session: pd.DataFrame) -> str:
    query = session.iloc[0]["query_text"]
    serp = "\t".join(session["url"].astype(str))

    return f"{session_id}\t0\tQ\t{query}\t0\t{serp}"


def create_click_record(session_id: str, row: pd.Series) -> str:
    view_time = row["total_view_time"] / 3600 if row["total_view_time"] > 0 else 0
    return f"{session_id}\t{view_time}\tC\t{row['url']}"


def click_target(row: pd.Series) -> bool:
    return row["is_click"]
    # return row["total_view_time"] > 60


def convert_df(
    path_from: Path, path_to: Path, target: Callable[[pd.Series], bool] = click_target
) -> None:
    # import matplotlib.pyplot as plt
    # from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(path_from, sep="\t")

    # scaler = MinMaxScaler()

    # print(df[df["total_view_time"] > 0.0]["total_view_time"].values.reshape(-1, 1))

    # scaler.fit(df[df["total_view_time"] > 0.0]["total_view_time"].values.reshape(-1, 1))

    # df["total_view_time"] = df["total_view_time"].apply(
    # lambda time: scaler.transform([[time]])[0][0] if time > 0 else time
    # )

    # query_counts = (
    # df[["session_id", "query_text"]].drop_duplicates().groupby("query_text").size()
    # )
    # valid_queries = query_counts[query_counts > 3].index

    # df = df[df["query_text"].isin(valid_queries)].reset_index(drop=True)

    # df_ = df[df["total_view_time"] > 0]
    # df_ = df_.loc[df_.groupby("session_id")["serp_pos"].idxmax()].reset_index(drop=True)

    # df["total_view_time"] = df["total_view_time"] / 60

    # df_["total_view_time"] = df_["total_view_time"].apply(
    # lambda t: t if t < 0 else np.log(t)
    # )

    # ax = df[(df["total_view_time"] > 0) & (df["total_view_time"] < 10)][
    # "total_view_time"
    # ].hist(bins=500)
    # ax = df_[df_["total_view_time"] < 3600]["total_view_time"].hist(bins=300)
    # fig = ax.get_figure()
    # fig.savefig("./hist.png")

    with open(path_to, "w", encoding="utf-8") as f_out:
        for session_id, session in df.groupby("session_id"):
            f_out.write(create_search_record(session_id, session))
            f_out.write("\n")

            for _, row in session.iterrows():
                if not target(row):
                    continue

                f_out.write(create_click_record(session_id, row))
                f_out.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser("convert")
    parser.add_argument(
        "--source", type=str, default=None, help="path to source sessions tsv."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="path to result sessions in .txt format.",
    )

    args = parser.parse_args()

    convert_df(args.source, args.output)
