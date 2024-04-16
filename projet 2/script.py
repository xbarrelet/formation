import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.frame import DataFrame

DATA_FILEPATH = 'resources/products.csv'

# This column has 32,792% of its values filled
TARGET_COLUMN = "categories_fr"

# These columns all have >50% of their values filled
CONSIDERED_COLUMNS = ["product_name", "brands", "categories_fr", "main_category_fr", "ingredients_text", "additives",
                      "nutrition_grade_fr",
                      "energy_100g"]  # For the outliers detection

# Pandas configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sns.set_theme()


def get_dtype(column_name: str) -> str:
    if column_name.endswith("_t"):
        return "timedelta64"
    elif column_name.endswith("_datetime"):
        return "datetime64"
    elif column_name.endswith("_100g"):
        return "float64"
    else:
        return "str"


def load_data_using_dtype_setting(nrows: int) -> DataFrame:
    columns_name = pd.read_csv(DATA_FILEPATH, sep="	", nrows=0).columns

    dtypes = {}
    converters = {}
    for column_name in columns_name:
        column_type = get_dtype(column_name)

        if column_type == "datetime64":
            converters[column_name] = lambda x: pd.to_datetime(x, errors='coerce', format='ISO8601')
        elif column_type == "timedelta64":
            converters[column_name] = lambda x: pd.to_timedelta(x, errors="coerce")
        elif column_name.endswith("_tags"):
            converters[column_name] = lambda x: str(x).split(',') if isinstance(x, str) else []
        else:
            dtypes[column_name] = column_type

    return pd.read_csv(DATA_FILEPATH, header=0, sep="	", nrows=nrows, dtype=dtypes, converters=converters)


def load_data(nrows: int) -> DataFrame:
    # dataframe = load_data_using_dtype_guessing(nrows)
    dataframe = load_data_using_dtype_setting(nrows)
    print("Data successfully loaded.\n")

    return dataframe


def load_data_using_dtype_guessing(nrows: int) -> DataFrame:
    dataframe = pd.read_csv(DATA_FILEPATH, header=0, sep="	", nrows=nrows, low_memory=False)
    columns_name = pd.read_csv(DATA_FILEPATH, sep="	", nrows=0).columns

    for column_name in columns_name:
        if column_name.endswith("_t"):
            dataframe[column_name] = pd.to_timedelta(dataframe[column_name], unit='s', errors='coerce')
        elif column_name.endswith("_datetime"):
            dataframe[column_name] = pd.to_datetime(dataframe[column_name], errors='coerce', format='ISO8601')
        elif column_name.endswith("_tags"):
            dataframe[column_name] = dataframe[column_name].apply(
                lambda x: x[1:-1].split(',') if isinstance(x, str) else [])

    return dataframe


def display_type_of_each_column(dataframe: DataFrame) -> None:
    print(dataframe.dtypes)
    print("\n")


def display_all_nan_values_of_column(dataframe: DataFrame, column_name: str) -> None:
    for item in dataframe[column_name]:
        if pd.notna(item):
            print(item)
    print("\n")


def display_information(dataframe: DataFrame) -> None:
    print(dataframe.info())
    print("\n")

    # display_type_of_each_column(dataframe)
    display_present_data_percentages(dataframe)
    # display_all_nan_values_of_column(dataframe, "carbon-footprint_100g")


def display_present_data_percentages(dataframe: DataFrame) -> None:
    present_data_percentages = dataframe.notna().mean().sort_values(ascending=False)

    print("Listing present data percentages for each column:")
    print(present_data_percentages)
    print("\n")


def isolate_quantitative_data(dataframe: DataFrame) -> DataFrame:
    return dataframe.select_dtypes(
        [np.number, 'datetime64'])  # TODO: datetime64 doesn't work but should? datetime64[ns, UTC](2)


def isolate_qualitative_data(dataframe: DataFrame) -> DataFrame:
    return dataframe.select_dtypes([np.object_])


def remove_duplicates(dataframe: DataFrame) -> None:
    initial_count = len(dataframe)
    pd.DataFrame.drop_duplicates(dataframe, subset=['product_name', 'quantity', 'brands'], inplace=True)
    duplicates_number = initial_count - len(dataframe)
    print(f"{duplicates_number} duplicates were removed based on the product name, brand and quantity columns.\n")


def load_and_filter_data(nrows: int = 10000000) -> DataFrame:
    dataframe: DataFrame = load_data(nrows)

    # df = isolate_quantitative_data(df)
    # df = isolate_qualitative_data(df)

    remove_duplicates(dataframe)
    return dataframe[CONSIDERED_COLUMNS]


def visualize_data_on_column(dataframe, column_name):
    print_the_outliers_values(column_name, dataframe)
    # show_distribution_plots(column_name, dataframe)


def show_distribution_plots(column_name, dataframe):
    sns.displot(data=dataframe, x=column_name)
    sns.catplot(data=dataframe, kind="swarm", x=column_name)
    plt.show()


def print_the_outliers_values(column_name, dataframe):
    filtered_dataframe = dataframe[dataframe[column_name].notna()]
    filtered_dataframe = filtered_dataframe[column_name]

    outliers_dataframe = extract_outliers(filtered_dataframe)

    print(f"Here are the outliers value in the column:{column_name}")
    print(outliers_dataframe)
    print("\n")


def extract_outliers(filtered_dataframe):
    Q1 = filtered_dataframe.quantile(0.25)
    Q3 = filtered_dataframe.quantile(0.75)
    IQR = Q3 - Q1
    mask = (filtered_dataframe < Q1 - 1.5 * IQR) | (filtered_dataframe > Q3 + 1.5 * IQR)
    filtered_dataframe = filtered_dataframe[mask]
    return filtered_dataframe


def visualize_data(dataframe: DataFrame) -> None:
    visualize_data_on_column(dataframe, "energy_100g")


if __name__ == '__main__':
    print("Welcome welcome! Let's prepare some data :)\n")

    # df: DataFrame = load_and_filter_data()
    df: DataFrame = load_and_filter_data(nrows=10000)
    data_with_no_missing_target_values = df[df[TARGET_COLUMN].notna()]

    # display_information(df)

    visualize_data(df)
