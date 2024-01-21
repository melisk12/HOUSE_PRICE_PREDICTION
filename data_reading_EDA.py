import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataframe_reading(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe

def check_data(dataframe):
    """
    Display key information about a DataFrame, including shape, head, tail, missing values, and descriptive statistics.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to be checked.

    Returns:
    This function prints the following information:
    1. Information: General information about the DataFrame.
    2. Data Shape: Shape of the DataFrame (number of rows and columns).
    3. The First 5 Data: Display the first 5 rows of the DataFrame.
    4. The Last 5 Data: Display the last 5 rows of the DataFrame.
    5. Missing Values: Display the count of missing values in each column.
    6. Describe the Data: Display basic statistical information about the DataFrame,
       including mean, standard deviation, minimum, maximum, and specified percentiles.

    Example:
    >>> check_data(df)
    """
    print(20 * "-" + "Information".center(20) + 20 * "-")
    print(dataframe.info())
    print(20 * "-" + "Data Shape".center(20) + 20 * "-")
    print(dataframe.shape)
    print("\n" + 20 * "-" + "The First 5 Data".center(20) + 20 * "-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print((dataframe.isnull().sum()).sort_values(ascending=False))
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    The function provides the names of categorical, numeric, and categorical but cardinal variables in the dataset.
    Note: Numeric-looking categorical variables are also included in the categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The desired dataframe to retrieve variable names
        cat_th: int, optional
                The class threshold value for variables that are numeric but categorical.
        car_th: int, optinal
                The class threshold value for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of categorical-looking cardinal variables.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is in cat_cols.
        The sum of 3 lists with return is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = number of variables
    # num_but_cat is already in cat_cols.
    # therefore, all variables will be selected with the following 3 lists: cat_cols + num_cols + cat_but_car
    # num_but_cat is provided for reporting only.

    return cat_cols, cat_but_car, num_cols

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    """
    Generate summary statistics for a numerical column in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - numerical_col (str): The name of the numerical column for which to generate summary statistics.
    - plot (bool, optional): Whether to plot a histogram of the numerical column (default is False).

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

def cat_summary(dataframe, col_name, plot=False):
    """
    Generate summary statistics for a categorical column in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - col_name (str): The name of the categorical column for which to generate summary statistics.
    - plot (bool, optional): Whether to plot a countplot of the categorical column (default is False).

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=45)
        plt.show()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def string_to_numerical(dataframe, overwrite=False):
    """
    Convert mistyped columns in a DataFrame to float type.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing columns to be converted.
    - overwrite (bool, optional): If True, overwrite the original DataFrame with converted values.

    Returns:
    None

    Prints:
    - The number of mistyped columns found and the corresponding column names.
    - If overwrite is True, also prints the number of columns converted and the names of mistyped columns.
    """
    numeric_cols = []
    converted_cols = {}

    for col in dataframe.columns:
        try:
            if dataframe[col].dtype != "O":
                numeric_cols.append(col)
            elif col not in numeric_cols:
                converted_cols[col] = pd.to_numeric(dataframe[col].replace(" ", ""))
        except ValueError:
            pass

    if overwrite:
        for col_name, converted_values in converted_cols.items():
            dataframe[col_name] = converted_values
        print(f"0 mistyped column(s) needs to be converted to float type.\n"
              f"{len(converted_cols.keys())} mistyped column(s) converted to float type.\n"
              f"Mistyped columns: {list(converted_cols.keys())}")
    else:
        print(f"{len(converted_cols)} mistyped column(s) needs to be converted to float type.")
        if len(converted_cols) > 0:
            print(f"Mistyped columns: {list(converted_cols.keys())}")
        else:
            print(f"No mistyped columns found.")