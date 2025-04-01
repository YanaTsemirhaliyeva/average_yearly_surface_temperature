import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("outputs/model_analysis.log"),
        logging.StreamHandler()
    ]
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ModelAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self):
        """Загрузка и предобработка данных"""
        try:
            logging.info("Загрузка данных...")
            self.data = pd.read_csv(self.file_path)
            self.data = self.data.rename(columns={
                'Entity': 'Country',
                'year': 'Year',
                'Average surface temperature': 'Daily average temp',
                'Average surface temperature.1': 'Yearly average temp'
            })
            self.data = self.data.drop(columns=['Code'])
            self.data['Day'] = pd.to_datetime(self.data['Day'])
            self.data['Month'] = self.data['Day'].dt.month

            # Добавление лагов и целевой переменной
            self.data = self.data.sort_values(by=["Country", "Day"])
            for lag in range(1, 8):
                self.data[f'Temp_Lag_{lag}'] = self.data.groupby('Country')['Daily average temp'].shift(lag)
            self.data['Target'] = self.data.groupby('Country')['Daily average temp'].shift(-1)
            self.data = self.data.dropna()

            # Разделение данных
            X = self.data.drop(columns=['Country', 'Day', 'Daily average temp', 'Yearly average temp', 'Target'])
            y = self.data['Target']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Масштабирование
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

            y_scaler = MinMaxScaler()
            self.y_train = y_scaler.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
            self.y_test = y_scaler.transform(self.y_test.values.reshape(-1, 1)).ravel()

            logging.info("Данные успешно загружены и предобработаны!")
        except Exception as e:
            logging.error(f"Ошибка при загрузке и предобработке данных: {e}")

    def evaluate_model(self, y_true, y_pred, model_name):
        """Оценка модели"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        logging.info(f"=== {model_name} ===")
        logging.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def train_and_evaluate_models(self):
        """Обучение и оценка моделей"""
        try:
            logging.info("Обучение моделей...")

            # Линейная регрессия
            lr_model = LinearRegression()
            lr_model.fit(self.X_train, self.y_train)
            y_pred_lr = lr_model.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_lr, "Linear Regression")

            # CatBoost
            catboost_model = CatBoostRegressor(verbose=0, random_seed=42)
            catboost_model.fit(self.X_train, self.y_train)
            y_pred_catboost = catboost_model.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_catboost, "CatBoost")

            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(self.X_train, self.y_train)
            y_pred_rf = rf_model.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_rf, "Random Forest")

            # LightGBM
            lgb_model = lgb.LGBMRegressor(random_state=42)
            lgb_model.fit(self.X_train, self.y_train)
            y_pred_lgb = lgb_model.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_lgb, "LightGBM")

            # Многослойный перцептрон
            mlp_model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            mlp_model.fit(self.X_train, self.y_train)
            y_pred_mlp = mlp_model.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_mlp, "MLPRegressor")

            # Ансамбль моделей
            voting_regressor = VotingRegressor(estimators=[
                ('lr', lr_model),
                ('catboost', catboost_model),
                ('rf', rf_model),
                ('lgb', lgb_model),
                ('xgbr', XGBRegressor(random_state=42))
            ])
            voting_regressor.fit(self.X_train, self.y_train)
            y_pred_voting = voting_regressor.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_voting, "Voting Ensemble")

            logging.info("Обучение моделей завершено!")
        except Exception as e:
            logging.error(f"Ошибка при обучении моделей: {e}")