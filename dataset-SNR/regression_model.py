from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression, RFECV
import warnings
import statistics as st
warnings.filterwarnings("ignore")


dataset = pd.read_csv("/Users/guilhermeribeiro/MEGA/PyCharmProjects/bacteria-cnpq/dataset-SNR/artifacts/graph_statistics.csv", encoding='utf-8')
dataset = dataset.replace(np.nan, 0)

# Remoção de outliers (SNRs >= 5)
dataset = dataset[dataset['Y']<5] # Remoção de outliers (SNRs >= 5)

labels = dataset[['Y']]
dataset = dataset.drop(['snr','Y', 'gate'], axis=1)


def rfe_svm_f1(dataset, labels):
    features = []
    selector_f = RFECV(LinearSVC(),  scoring='f1_weighted')

    selector_f.fit(dataset, labels)
    features.append(list(dataset.columns.values[selector_f.support_]))

    features_s = [x for l in features for x in l]
    features_df = pd.DataFrame(features_s, columns=['features'])
    features_df['value'] = 1
    features_df = features_df.groupby(by='features').count().sort_values(by='value', ascending=False)
    features_df.reset_index(inplace=True)

    return features_df.loc[0:7, 'features']

mae_l = []
mse_l = []
rmse_l = []

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.33,
                                                        random_state=i)

    # 3. Initialize and fit the model
#     model = Pipeline([
#     ("scaler", StandardScaler()),
#     ("regressor", SGDRegressor(
#         loss="squared_error",
#         penalty="l2",
#         alpha=0.0001,
#         learning_rate="adaptive",
#         eta0=0.001,
#         max_iter=5000,
#         tol=1e-4,
#         early_stopping=True,
#         validation_fraction=0.1,
#         n_iter_no_change=20,
#         random_state=42
#     ))
# ])
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # 4. Make predictions
    y_pred = model.predict(X_test)

    # 1. Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    mae_l.append(mae)

    # 2. Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_l.append(mse)

    # 3. Root Mean Squared Error (RMSE) - Supported in modern scikit-learn versions
    rmse = root_mean_squared_error(y_test, y_pred)
    rmse_l.append(rmse)

    print(f"run: {i}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")


print(f"Final (AVG) - MAE: {st.mean(mae_l)}, MSE: {st.mean(mse_l)}, RMSE:"
      f" {st.mean(rmse_l)}")