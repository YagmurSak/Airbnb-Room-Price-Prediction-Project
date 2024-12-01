import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("TkAgg")
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.impute import KNNImputer

import joblib
from warnings import filterwarnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("TkAgg")
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.impute import KNNImputer

import joblib
from warnings import filterwarnings
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

filterwarnings('ignore')


def train_and_save_model():
    df = pd.read_csv("AB_NYC_2019.csv")

    def data_prep(dataframe):
        dataframe.drop(["name", "host_id", "host_name", "last_review", "id","latitude","longitude"], axis=1, inplace=True, errors='ignore')
        return dataframe

    df = data_prep(df)

    def grab_col_names(dataframe, cat_th=10, car_th=20):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if
                       dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if
                       dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)} {cat_cols}')
        print(f'num_cols: {len(num_cols)} {num_cols}')
        print(f'cat_but_car: {len(cat_but_car)} {cat_but_car}')
        print(f'num_but_cat: {len(num_but_cat)} {num_but_cat}')
        return cat_cols, num_cols, cat_but_car

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
        quantile_one = dataframe[variable].quantile(low_quantile)
        quantile_three = dataframe[variable].quantile(up_quantile)
        interquantile_range = quantile_three - quantile_one
        up_limit = quantile_three + 1.5 * interquantile_range
        low_limit = quantile_one - 1.5 * interquantile_range
        return low_limit, up_limit

    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

    for col in num_cols:
        if col != "price":
            print(col, check_outlier(df, col))

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    for col in num_cols:
        replace_with_thresholds(df, col)

    def missed(df):
        dff = df[["reviews_per_month"]]
        rs = RobustScaler()
        dff = pd.DataFrame(rs.fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(rs.inverse_transform(dff), columns=dff.columns)
        df[["reviews_per_month"]] = dff
        return df

    df = missed(df)

    def rare_analyser(dataframe, target, cat_cols):
        for col in cat_cols:
            print(col, ':', len(dataframe[col].value_counts()))
            print(pd.DataFrame({'COUNT': dataframe[col].value_counts(),
                                'RATIO': dataframe[col].value_counts() / len(dataframe),
                                'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')

    rare_analyser(df, "price", cat_cols)

    def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()
        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

        for var in rare_columns:
            tmp = temp_df[var].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        return temp_df

    df = rare_encoder(df, 0.01)

    def create_new_features(df):
        # df['NEW_total_review_ratio'] = df['number_of_reviews'] / (df['availability_365'] + 1)
        # df['NEW_average_income_per_review'] = (df['price'] * df['availability_365']) / (df['number_of_reviews'] + 1)
        # df['NEW_average_nights_booked'] = df['availability_365'] / (df['number_of_reviews'] + 1)
        # df['NEW_price_per_review'] = df['price'] / (df['number_of_reviews'] + 1)
        # df['NEW_days_since_last_review'] = (365 - df['availability_365']) / (df['reviews_per_month'] + 1)
        # df['NEW_monthly_income_estimate'] = (df['price'] * df['availability_365']) / 12
        # df['NEW_review_density'] = df['number_of_reviews'] / 365
        # df['NEW_total_cost'] = df['price'] * df['minimum_nights']
        # df['NEW_annual_income'] = df['price'] * df['availability_365']
        # This feature represents the total cost of the house for the minimum number of nights. It takes the total of the price for minimum nights.
        # df['NEW_total_cost'] = df['price'] * df['minimum_nights']

        # This feature can be used to estimate for how long a house has been listed. This duration is calculated by dividing the total number of reviews that the house has received by the number of reviews per month.
        df['NEW_estimated_listed_months'] = df['number_of_reviews'] / df['reviews_per_month']

        # This feature gives the ratio of how long a house is available throughout the year.
        df['NEW_availability_ratio'] = df['availability_365'] / 365

        # This feature gives the daily average reviews a host receives. It divides the reviews per month by the number of days in a month.
        df['NEW_daily_average_reviews'] = df['reviews_per_month'] / 30

        # This feature estimates how much a host can earn in a year. It multiplies the price of a house with how many days it is available in a year.
        # df['NEW_annual_income'] = df['price'] * df['availability_365']

        # This feature estimates the average duration a customer stays. It divides the total number of reviews by the reviews per month.
        df['NEW_average_stay_duration'] = df['number_of_reviews'] / df['reviews_per_month']

        # This feature gives the occupancy rate of a house throughout the year. It subtracts from 365 the number of days a house is available in a year.
        df['NEW_house_occupancy_rate'] = 365 - df['availability_365']

        # This feature determines the minimum amount a house can get for a booking. It multiplies the price of a house with the minimum nights.
        # df['NEW_minimum_income'] = df['price'] * df['minimum_nights']
        return df

    df = create_new_features(df)

    #def scale_numeric_columns(dataframe, numeric_columns, exclude_columns=None):
    #    if exclude_columns:
    #        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

     #   rs = RobustScaler()
     #   dataframe[numeric_columns] = rs.fit_transform(dataframe[numeric_columns])
      #  return dataframe[numeric_columns].head()
       # # TODO: return dataframe[numeric_columns].head()

    #num_cols = [col for col in df.select_dtypes(include="number").columns]
    #scaled_data = scale_numeric_columns(df, num_cols, exclude_columns=["price"])
    #print(scaled_data)

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Save the processed DataFrame to a new CSV file
    df.to_csv("yeni_dosya1.csv", index=False)
    print("yeni_dosya1.csv başarıyla oluşturuldu!")

    # Verinin ayrıştırılması
    y = df["price"]

    # Hedef değişken ve kategorik bir kolon çıkarılıyor
    X = df.drop(["neighbourhood", "price"], axis=1)

    # Eğitim ve test veri setlerine bölünmesi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

    # CatBoost modelinin oluşturulması
    model = CatBoostRegressor(random_state=42, verbose=200, iterations=500, learning_rate=0.1, depth=6)

    # Modelin eğitilmesi
    model.fit(X_train, y_train, cat_features=[])

    # Tahminler
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

    # Modelin kaydedilmesi
    joblib.dump(model, "yeni_cikti_catboost.pkl")
    print("Model başarıyla 'yeni_cikti_catboost.pkl' olarak kaydedildi.")


train_and_save_model()
