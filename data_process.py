import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _bin_age(age_series):
    bins = [-np.inf, 10, 40, np.inf]
    labels = ["Child", "Adult", "Elderly"]
    return (
        pd.cut(age_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


def _extract_title(name_series):
    titles = name_series.str.extract(" ([A-Za-z]+)\.", expand=False)
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


def _create_features(df):
    # Convert 'Age' to numeric, coercing errors to NaN
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = _bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
    all_keywords = set(df.columns)
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )
    return df, all_keywords


def get_partitions_and_label(partition_count=3, data_type="vanilla"):
    assert data_type == "vanilla" and partition_count == 3
    assert data_type == "vanilla" or data_type == "custom"

    if data_type == "vanilla":
        df = pd.read_csv("_static/data/train.csv")
        processed_df = df.dropna(subset=["Embarked", "Fare"]).copy()
        processed_df, all_keywords = _create_features(processed_df)

        raw_partitions = _partition_data(processed_df, all_keywords)
        partitions = []
        for partition in raw_partitions:
            partitions.append(partition.drop("Survived", axis=1))
        return partitions, processed_df["Survived"].values

    elif data_type == "custom":
        print("Not Implemented yet!")
        return 0, 0


def _partition_data(df, all_keywords):
    partitions = []
    keywords_sets = [{"Parch", "Cabin", "Pclass"}, {"Sex", "Title"}]
    keywords_sets.append(all_keywords - keywords_sets[0] - keywords_sets[1])

    for keywords in keywords_sets:
        partitions.append(
            df[
                list(
                    {
                        col
                        for col in df.columns
                        for kw in keywords
                        if kw in col or "Survived" in col
                    }
                )
            ]
        )

    return partitions


def data_transform(data):
    return StandardScaler().fit_transform(data)


def gen_batches(data_num, batch_size):
    batch_num = data_num // batch_size
    size_list = [batch_size] * batch_num
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    res = list()
    b = 0
    for size in size_list:
        res.append(indexes[b : b + size])
        b += size
    return res
