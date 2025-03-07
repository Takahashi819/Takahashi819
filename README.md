<!---
Takahashi819/Takahashi819 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

# 解析コード
## GPPコード
## XGBoost
#### ライブラリインポート
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from sklearn.model_selection import PredefinedSplit

#### データの読み込みと前処理
din_asiaflux = np.loadtxt("/home/test/research/forest.csv", delimiter=",", skiprows=1)
sw = din_asiaflux[:,2]
lai = din_asiaflux[:,5]
ave = din_asiaflux[:,6]
amp = din_asiaflux[:,7]
lst_day = din_asiaflux[:,8]
lst_night = din_asiaflux[:,9]
brdf1 = din_asiaflux[:,10]
brdf2 = din_asiaflux[:,11]
brdf3 = din_asiaflux[:,12]
brdf4 = din_asiaflux[:,13]
brdf5 = din_asiaflux[:,14]
brdf6 = din_asiaflux[:,15]
brdf7 = din_asiaflux[:,16]
lc = din_asiaflux[:,17]
ndvi = (brdf2 - brdf1) / (brdf2 + brdf1)  
evi = 2.5 * (brdf2 - brdf1) / (brdf2 + 6 * brdf1 - 7.5 * brdf3 + 1)
lswi = (brdf2 - brdf6) / (brdf2 + brdf6)
ndwi = (brdf2 - brdf5) / (brdf2 + brdf5)
obs_gpp = din_asiaflux[:,4]

#### --- 特徴量セットの定義 ---
feature_combinations = [
    (sw, lst_day, lai, ndvi, lswi, lc),
    (sw, lst_day, lai, ndvi, lswi, ave, amp),
    (sw, lst_day, lai, evi, lswi, lc),
    (sw, lst_day, lai, evi, lswi, ave, amp),
    (sw, lst_day, lai, ndvi, ndwi, lc),
    (sw, lst_day, lai, ndvi, ndwi, ave, amp),
    (sw, lst_day, lai, evi, ndwi, lc),
    (sw, lst_day, lai, evi, ndwi, ave, amp),
    (sw, lst_day, lc, brdf1, brdf2, brdf3, brdf4, brdf5, brdf6, brdf7),
    (sw, lst_day, ave, amp, brdf1, brdf2, brdf3, brdf4, brdf5, brdf6, brdf7)
]

#### カスタム分割インデックスを作成
split_dict = {
    1: [(2379, 2516), (6410, 6896), (7838, 8007)],
 2: [(1519, 1991), (2517, 2653), (6897, 6987), (7746, 7837)],
 3: [(1992, 2082), (2974, 3018), (4453, 4863), (6988, 7065)],
 4: [(2083, 2195), (2654, 2973), (4864, 5229), (7066, 7202), (8008, 8272)],
 5: [(0, 11), (2196, 2378), (7426, 7745)],
 6: [(12, 142), (3019, 3467), (5632, 5952), (7294, 7425)],
 7: [(143, 305), (3468, 3857), (5953, 6226)],
 8: [(306, 428), (1345, 1518), (5230, 5631)],
 9: [(499, 1344), (6227, 6409)],
 10: [(429, 498), (3858, 4452), (7203, 7293)]
}

#### データ長を取得
num_samples = len(obs_gpp)
custom_fold_indices = np.full(num_samples, -1)

for fold, ranges in split_dict.items():
    for start, end in ranges:
        custom_fold_indices[start:end + 1] = fold - 1

custom_split = PredefinedSplit(custom_fold_indices)

param_grid = {
    'n_estimators': [500, 800, 1200],  
    'max_depth': [5, 8, 12],  
    'learning_rate': [0.005, 0.01, 0.1],  
    'subsample': [0.8],  
    'colsample_bytree': [0.7],  
    'reg_alpha': [0.5, 1],  
    'reg_lambda': [1.0, 2.0],  
    'gamma': [0.5, 2, 5]  
}

best_models = []
best_params_per_fold = {}

#### 特徴量の1つ目でグリッドサーチ
X = np.column_stack(feature_combinations[0])
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

#### 各foldごとに個別にパラメータを最適化
for fold in range(np.max(custom_fold_indices) + 1):  # fold は 0-indexed
    print(f"Processing fold {fold + 1}...")

    train_idx = np.where((custom_fold_indices != fold) & (custom_fold_indices >= 0))[0]
    valid_idx = np.where(custom_fold_indices == fold)[0]

    if valid_idx.size == 0:
        print(f"Warning: No validation data for fold {fold + 1}")
        continue

    X_train, X_valid = X_normalized[train_idx], X_normalized[valid_idx]
    y_train, y_valid = obs_gpp[train_idx], obs_gpp[valid_idx]

    # `train_idx` に対応した `PredefinedSplit`
    fold_split = PredefinedSplit(custom_fold_indices[train_idx])

    # XGB モデルの作成
    xgb = XGBRegressor()

    # GridSearchCV 
    grid_search = GridSearchCV(xgb, param_grid, scoring='neg_mean_squared_error', cv=fold_split, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 最適なモデルを保存
    best_model = grid_search.best_estimator_
    best_models.append(best_model)

    # 最適なパラメータを保存
    best_params_per_fold[fold] = grid_search.best_params_
    print(f"Best Parameters for fold {fold + 1}: {grid_search.best_params_}")

#### --- 各foldの最適なパラメータ表示 ---
print("Best Parameters for each fold:")
for fold, params in best_params_per_fold.items():
    print(f"Fold {fold + 1}: {params}")

#### --- クロスバリデーション予測 ---
ensemble_predictions = []
for fold in range(len(best_models)):
    y_pred = best_models[fold].predict(X_normalized)
    ensemble_predictions.append(y_pred)

#### --- アンサンブル平均の計算 ---
ensemble_predictions = np.array(ensemble_predictions)
ensemble_mean = np.mean(ensemble_predictions, axis=0)

print("Cross-validation with per-fold parameter optimization is completed.")

#### 線形回帰モデルのトレーニングと予測
reg_model = LinearRegression()
reg_model.fit(obs_gpp.reshape(-1, 1), ensemble_mean)
predicted_line = reg_model.predict(obs_gpp.reshape(-1, 1))

#### 濃度計算
xy = np.vstack([obs_gpp, ensemble_mean])
z = gaussian_kde(xy)(xy)

#### MSE, RMSE, MBE の再計算
r_correlation = np.corrcoef(obs_gpp, ensemble_mean)[0, 1]  
r2_regression = r_correlation ** 2  
mse_regression = mean_squared_error(obs_gpp, ensemble_mean)
rmse_regression = np.sqrt(mse_regression)
mbe_regression = np.mean(ensemble_mean - obs_gpp)
data_count = len(obs_gpp)

plt.figure(figsize=(6, 6))
plt.scatter(obs_gpp, ensemble_mean, alpha=0.6, color='blue', s=10)
plt.scatter(obs_gpp, ensemble_mean, c=z, cmap='jet', alpha=0.6, s=10)
plt.plot(obs_gpp, predicted_line, color='red', linewidth=2,
         label=f"forest\n"
               f"data={data_count}\n"
               f"R²={r2_regression:.2g} \n"
               f"RMSE={rmse_regression:.2g}\n"
               f"MBE={mbe_regression:.2g}\n"
               f"y = {reg_model.coef_[0]:.2g}x + {reg_model.intercept_:.2g}")

plt.legend(fontsize=16, loc='upper right')
plt.xlabel(r"Obs GPP(gC m$^{-2}$ day$^{-1}$)", fontsize=20)
plt.ylabel("Model GPP(gC m$^{-2}$ day$^{-1}$)", fontsize=20)
plt.text(1, 26, "XGBoost", fontsize=24, ha='left', va='top', color='black')
plt.xlim(-2.5, 30)
plt.ylim(-2.5, 30)
plt.gca().set_aspect('equal', adjustable='box')

plt.grid(False)
plt.show()

#### --- CSVファイル保存 ---
output_path = "/home/test/predict/forest_xgb_results.csv"

#### DataFrame作成
results_df = pd.DataFrame({
    "Obs_GPP": obs_gpp,
    "Predicted_GPP": ensemble_mean
})

results_df.to_csv(output_path, index=False)

print(f"予測結果をCSVファイルに保存しました: {output_path}")
print(f"回帰直線の式: y = {reg_model.coef_[0]:.2g}x + {reg_model.intercept_:.2g}")
print(f"RMSE: {rmse_regression:.2g}")
print(f"R² (相関係数²): {r2_regression:.2g}")
print(f"MBE: {mbe_regression:.2g}")

### 非森林、全サイト
#### 非森林データ
din_asiaflux = np.loadtxt("/home/test/research/nonforest.csv", delimiter=",", skiprows=1)
#### 非森林カスタム分割インデックス
split_dict = {
    1: [(0, 128), (391, 416)],
 2: [(129, 197), (1122, 1396)],
 3: [(198, 373), (2258, 2500)],
 4: [(374, 390)],
 5: [(782, 854), (1397, 1887)],
 6: [(417, 492)],
 7: [(493, 517), (1888, 1979)],
 8: [(855, 991), (1980, 2157)],
 9: [(992, 1121), (2158, 2257)],
 10: [(518, 781)]
}

#### 全サイトデータ
din_asiaflux = np.loadtxt("/home/test/research/all_sites.csv", delimiter=",", skiprows=1)
#### 全サイトカスタム分割インデックス
split_dict = {
    1: [(0, 128), (1576, 1601), (3161, 3298), (8298, 8784), (10096, 10265)],
 2: [(129, 197),
  (2301, 2773),
  (3299, 3435),
  (5575, 5849),
  (8785, 8875),
  (10004, 10095)],
 3: [(198, 373),
  (2774, 2864),
  (3756, 3800),
  (5850, 6260),
  (8876, 8953),
  (10266, 10508)],
 4: [(374, 390),
  (2865, 2977),
  (3436, 3755),
  (6261, 6626),
  (8954, 9090),
  (10509, 10773)],
 5: [(391, 402), (2978, 3160), (4640, 4712), (7624, 8114), (9684, 10003)],
 6: [(403, 533), (1602, 1677), (3801, 4249), (7029, 7349), (9182, 9313)],
 7: [(534, 696), (1678, 1702), (4250, 4639), (7350, 7623), (9314, 9405)],
 8: [(697, 819), (2127, 2300), (4713, 4849), (6627, 7028), (9406, 9583)],
 9: [(890, 1575), (1703, 1862), (4850, 4979), (8115, 8297), (9584, 9683)],
 10: [(820, 889), (1863, 2126), (4980, 5574), (9091, 9181)]
}

### SVR,RF
#### SVR
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.1, 0.5, 1],
    'gamma': [1, 3]
}

xgb = XGBRegressor()をsvr = SVR()に変更

#### RF
param_grid = {
    'n_estimators': [500, 700, 1000],
    'max_depth': [5, 8],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [2, 5, 8]
}

xgb = XGBRegressor()をrf = RandomForestRegressor(random_state=42)に変更


## NEEコード(変更点)
#### データの読み込みと前処理
din_asiaflux = np.loadtxt("/home/test/research/forest.csv", delimiter=",", skiprows=1)
sw = din_asiaflux[:,2]
lai = din_asiaflux[:,5]
ave = din_asiaflux[:,6]
amp = din_asiaflux[:,7]
lst_day = din_asiaflux[:,8]
lst_night = din_asiaflux[:,9]
brdf1 = din_asiaflux[:,10]
brdf2 = din_asiaflux[:,11]
brdf3 = din_asiaflux[:,12]
brdf4 = din_asiaflux[:,13]
brdf5 = din_asiaflux[:,14]
brdf6 = din_asiaflux[:,15]
brdf7 = din_asiaflux[:,16]
lc = din_asiaflux[:,17]
pre_gpp = din_asiaflux[:,18]
ndvi = (brdf2 - brdf1) / (brdf2 + brdf1)  
evi = 2.5 * (brdf2 - brdf1) / (brdf2 + 6 * brdf1 - 7.5 * brdf3 + 1)
lswi = (brdf2 - brdf6) / (brdf2 + brdf6)
ndwi = (brdf2 - brdf5) / (brdf2 + brdf5)
lst_ave = (lst_day + lst_night) / 2
obs_nee = din_asiaflux[:,3]

#### --- 特徴量セットの定義 ---
feature_combinations = [
    (sw, lst_ave, lai, ndvi, lswi, lc, pre_gpp),
    (sw, lst_ave, lai, ndvi, lswi, ave, amp, pre_gpp),
    (sw, lst_ave, lai, evi, lswi, lc, pre_gpp),
    (sw, lst_ave, lai, evi, lswi, ave, amp, pre_gpp),
    (sw, lst_ave, lai, ndvi, ndwi, lc, pre_gpp),
    (sw, lst_ave, lai, ndvi, ndwi, ave, amp, pre_gpp),
    (sw, lst_ave, lai, evi, ndwi, lc, pre_gpp),
    (sw, lst_ave, lai, evi, ndwi, ave, amp, pre_gpp),
    (sw, lst_ave, lc, brdf1, brdf2, brdf3, brdf4, brdf5, brdf6, brdf7, pre_gpp),
    (sw, lst_ave, ave, amp, brdf1, brdf2, brdf3, brdf4, brdf5, brdf6, brdf7, pre_gpp)
]
#### 残りの変更
変数obs_gppをobs_neeにする

# asiaflux061.csvからデータを読み取る
import numpy as np

#### データの読み込み（17列と10列）
data_column_17 = np.loadtxt("asia_flux_061.csv", delimiter=",", usecols=17, skiprows=1)
data_column_10 = np.loadtxt("asia_flux_061.csv", delimiter=",", usecols=10, skiprows=1)

#### -9999 を除いた行のインデックスを取得（17列と10列）
valid_rows_mask = (data_column_17 != -9999) & (data_column_10 != -9999)

#### 84列と85列のデータを読み込む(今回はndiv_ave,amp)
data_columns = [
    np.loadtxt("asia_flux_061.csv", delimiter=",", usecols=i, skiprows=1)
    for i in [84, 85]
]

#### 17列と10列のクリーンなデータを適用したマスクを使用
cleaned_data_columns = [col[valid_rows_mask] for col in data_columns]

#### クリーンデータを保存
np.savetxt(
    "cleaned_data_columns.csv",
    np.column_stack(cleaned_data_columns),
    delimiter=",",
    header="ave,amp",
    comments=""
)

#### クリーンなデータの総数を表示
total_cleaned_data = np.sum(valid_rows_mask)
print(f"クリーンなデータの総数: {total_cleaned_data}")

# サイトごとの平均
## NEEの行の抽出
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### データの読み込みと前処理
din_asiaflux = np.loadtxt("/home/test/research/analysis_data_set.csv", delimiter=",", skiprows=1)
ID = din_asiaflux[:, 0]
cross_ID = din_asiaflux[:, 1]
sw = din_asiaflux[:, 2]
nee = din_asiaflux[:, 3]
obs_gpp = din_asiaflux[:, 4]
lai = din_asiaflux[:, 5]
ave = din_asiaflux[:, 6]
amp = din_asiaflux[:, 7]
lst_day = din_asiaflux[:, 8]
lst_night = din_asiaflux[:, 9]
brdf1 = din_asiaflux[:, 10]
brdf2 = din_asiaflux[:, 11]
brdf3 = din_asiaflux[:, 12]
brdf4 = din_asiaflux[:, 13]
brdf5 = din_asiaflux[:, 14]
brdf6 = din_asiaflux[:, 15]
brdf7 = din_asiaflux[:, 16]
lc = din_asiaflux[:, 17]

#### DataFrame に変換
data = pd.DataFrame({'ID': ID, 'nee': nee})

#### ID ごとに gpp の平均を計算
grouped_mean = data.groupby('ID')['nee'].mean()
sw
#### 結果を表示
print(grouped_mean)
grouped_mean.to_csv("/home/test/ダウンロード/grouped_means.csv")

# サイトごとの解析
## Fyo(XGBoost)
#### データの読み込み
din_asiaflux = np.loadtxt("/home/test/research/forest.csv", delimiter=",", skiprows=1)
din_fyo = din_asiaflux[499:1185]
sw = din_fyo[:, 2]
nee = din_fyo[:, 3]
obs_gpp = din_fyo[:, 4]
lai = din_fyo[:, 5]
ave = din_fyo[:, 6]
amp = din_fyo[:, 7]
lst_day = din_fyo[:, 8]
lst_night = din_fyo[:, 9]
brdf1 = din_fyo[:, 10]
brdf2 = din_fyo[:, 11]
brdf3 = din_fyo[:, 12]
brdf4 = din_fyo[:, 13]
brdf5 = din_fyo[:, 14]
brdf6 = din_fyo[:, 15]
brdf7 = din_fyo[:, 16]
lc = din_fyo[:, 17]
ndvi = (brdf2 - brdf1) / (brdf2 + brdf1)
evi = 2.5 * (brdf2 - brdf1) / (brdf2 + 6 * brdf1 - 7.5 * brdf3 + 1)
lswi = (brdf2 - brdf6) / (brdf2 + brdf6)
ndwi = (brdf2 - brdf5) / (brdf2 + brdf5)

#### 特徴量セットの定義
feature_combinations = [
    (sw, lst_day, lai, ndvi, lswi, lc),
    (sw, lst_day, lai, ndvi, lswi, ave, amp),
    (sw, lst_day, lai, evi, lswi, lc),
    (sw, lst_day, lai, evi, lswi, ave, amp),
    (sw, lst_day, lai, ndvi, ndwi, lc),
    (sw, lst_day, lai, ndvi, ndwi, ave, amp),
    (sw, lst_day, lai, evi, ndwi, lc),
    (sw, lst_day, lai, evi, ndwi, ave, amp),
    (sw, lst_day, lc, brdf1, brdf2, brdf3, brdf4, brdf5, brdf6, brdf7),
    (sw, lst_day, ave, amp, brdf1, brdf2, brdf3, brdf4, brdf5, brdf6, brdf7)
]

#### --- KFold を使用して普通の10重クロスバリデーションに変更 ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scaler = MinMaxScaler()
xgb = XGBRegressor()

#### 普通の10重クロスバリデーションでの予測
ensemble_predictions = []  # 各特徴量セットの予測値を保存
for features in feature_combinations:
    #### 特徴量を結合
    X = np.column_stack(features)
    X_normalized = scaler.fit_transform(X)

    # クロスバリデーション予測
    y_pred = cross_val_predict(xgb, X_normalized, obs_gpp, cv=kf)
    ensemble_predictions.append(y_pred)

#### アンサンブル平均の計算
ensemble_predictions = np.array(ensemble_predictions)  # shape: (num_features, num_samples)
ensemble_mean_xgb = np.mean(ensemble_predictions, axis=0)  # サンプルごとの平均値

#### グリッドサーチ用のパラメータ
param_grid = {
    'n_estimators': [500, 800, 1200],  
    'max_depth': [5, 8, 12],  
    'learning_rate': [0.005, 0.01, 0.1],  
    'subsample': [0.8],  
    'colsample_bytree': [0.7],  
    'reg_alpha': [0.5, 1],  
    'reg_lambda': [1.0, 2.0],  
    'gamma': [0.5, 2, 5]  
}

#### 特徴量の1つ目でグリッドサーチ
X = np.column_stack(feature_combinations[0])  # 最初の特徴量セット
X_normalized = scaler.fit_transform(X)

#### グリッドサーチの実行
grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_normalized, obs_gpp)

#### 最適なパラメータとスコアを表示
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score (neg_mean_squared_error):", best_score)

#### 線形回帰モデルのトレーニングと予測
reg_model = LinearRegression()
reg_model.fit(obs_gpp.reshape(-1, 1), ensemble_mean_xgb)
predicted_line = reg_model.predict(obs_gpp.reshape(-1, 1))

#### MSEとR^2スコアの再計算
mse_regression = mean_squared_error(obs_gpp, ensemble_mean_xgb)
rmse_regression_xgb = np.sqrt(mse_regression)
r2_regression_xgb = r2_score(obs_gpp, ensemble_mean_xgb)
mbe_regression_xgb = np.mean(ensemble_mean_xgb - obs_gpp)
data_count = len(obs_gpp)







