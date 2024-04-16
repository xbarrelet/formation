import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

DATA_FILEPATH = 'resources/products.csv'

REQUIRED_COLUMNS = ["quantity"]

# Pandas configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def get_dtype(column_name: str):
    if column_name.endswith("_t"):
        return "timedelta64[s]"
    if column_name.endswith("_datetime"):
        return "datetime64"
    else:
        return "str"


def load_data(nrows=1000000) -> DataFrame:
    columns_name = pd.read_csv(DATA_FILEPATH, delim_whitespace=True, nrows=0).columns

    dtypes = {}
    columns_with_datetime_type = []
    converters = {}
    for column_name in columns_name:
        dtypes[column_name] = get_dtype(column_name)

        if dtypes[column_name] == "datetime64":
            columns_with_datetime_type.append(column_name)
        elif dtypes[column_name] == "timedelta64[s]":
            converters[column_name] = lambda x: pd.to_timedelta(x)

    dataframe = pd.read_csv(DATA_FILEPATH, header=0, sep="	",
                            nrows=nrows,
                            parse_dates=columns_with_datetime_type,
                            date_format='ISO8601',
                            converters=converters
                            )

    print("Data successfully loaded.\n")
    return dataframe


def display_type_of_each_column(dataframe):
    print(dataframe.dtypes)
    print("\n")


def display_information(dataframe: DataFrame):
    print(df.info())
    # print(dataframe.describe())
    # display_type_of_each_column(dataframe)
    # TODO: Use https://www.geeksforgeeks.org/python-visualize-missing-values-nan-values-using-missingno-library/
    # display_missing_data_percentages(dataframe)


def display_missing_data_percentages(dataframe):
    missing_data_percentages = dataframe.isna().mean().sort_values(ascending=False)
    print("Listing missing data percentages for each column:")
    print(missing_data_percentages)
    print("\n")


def retrieve_quantitative_data(df):
    return df.select_dtypes(
        [np.number, 'datetime64'])  # TODO: datetime64 doesn't work but should? datetime64[ns, UTC](2)


def retrieve_qualitative_data(df):
    return df.select_dtypes([np.object_])


def filter_dataframe(df):
    for column in REQUIRED_COLUMNS:
        df = df[df[column].notna()]
    return df


if __name__ == '__main__':
    print("Welcome welcome! Let's prepare some data :)\n")

    # Load the first 1000 lines for now
    df: DataFrame = load_data(nrows=1000)

    # df = retrieve_quantitative_data(df)
    # df = retrieve_qualitative_data(df)
    filtered_df = filter_dataframe(df)

    # Display information
    display_information(filtered_df)

    # TODO:
    # - Sélectionnez des features qui sont assez remplis (plus que 50%) et qui vous paraissent intéressantes pour effectuer la prédiction de votre cible.
    # - Supprimez les produits en double. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
