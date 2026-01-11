

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def load_and_prepare_data(
    products_path="data/products.csv",
    departments_path="data/departments.csv",
    test_size=0.2,
    random_state=42,
):

    df_products = pd.read_csv(products_path)
    df_departments = pd.read_csv(departments_path)

    # Merge on department_id
    df = pd.merge(df_products, df_departments, on="department_id")


    df["text"] = df.apply(
        lambda row: f"{row['product_name']} ->: {row['department']}",
        axis=1,
    )


    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )


    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )

    return dataset, train_df, test_df


if __name__ == "__main__":
    dataset, train_df, test_df = load_and_prepare_data()
    print(dataset)
