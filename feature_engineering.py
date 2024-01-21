import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    """
    Generate a table displaying the count and percentage of missing values for each column in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to analyze for missing values.
    - na_name (bool, optional): Whether to return a list of column names with missing values (default is False).

    Returns:
    None or list: If `na_name` is True, returns a list of column names with missing values; otherwise, prints the missing values table.

    Example:
    >>> missing_values_table(df)
         n_miss  ratio
    col3      15   7.50
    col1       5   2.50
    col2       3   1.50

    >>> missing_columns = missing_values_table(df, na_name=True)
    >>> print(f"Columns with missing values: {missing_columns}")
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    """
    Quickly impute missing values in a DataFrame based on specified methods.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing missing values to be imputed.
    - num_method (str, optional): The imputation method for numeric variables ('mean' or 'median', default is 'median').
    - cat_length (int, optional): The maximum number of unique values for a variable to be considered categorical (default is 20).
    - target (str, optional): The target variable used for imputing missing values in numeric columns (default is 'SalePrice').

    Returns:
    - pd.DataFrame: The DataFrame with missing values imputed.

    Example:
    >>> data = quick_missing_imp(data, num_method="mean", cat_length=15, target="Price")

    """
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def standart_scaler_func(dataframe, num_cols):
    ss = StandardScaler()
    dataframe[num_cols] = ss.fit_transform(dataframe[[num_cols]])
    return dataframe


def min_max_scaler_func(dataframe, num_cols):
    mms = MinMaxScaler()
    dataframe[num_cols] = mms.fit_transform(dataframe[[num_cols]])
    return dataframe


def robust_scaler_func(dataframe, num_cols):
    rs = RobustScaler()
    dataframe[num_cols] = rs.fit_transform(dataframe[[num_cols]])
    return dataframe

def scaling_func(dataframe, num_cols, name="robust"):
    """
    Apply scaling to numeric columns in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data to be scaled.
    - num_cols (list): A list of column names representing numeric variables to be scaled.
    - name (str, optional): The scaling method to use ('robust', 'min_max', or 'standard', default is 'robust').

    Returns:
    - pd.DataFrame: The DataFrame with scaled numeric columns.

    Example:
    >>> data = scaling_func(data, num_cols=['Age', 'Income'], name='min_max')
    """
    if name == "robust":
        rs = RobustScaler()
        dataframe[num_cols] = rs.fit_transform(dataframe[[num_cols]])
        return dataframe
    elif name == "min_max":
        mms = MinMaxScaler()
        dataframe[num_cols] = mms.fit_transform(dataframe[[num_cols]])
        return dataframe
    elif name == "standart":
        ss = StandardScaler()
        dataframe[num_cols] = ss.fit_transform(dataframe[[num_cols]])
        return dataframe