import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Логирование в файл
        logging.StreamHandler()         # Логирование в консоль
    ]
)

# Создание папки для сохранения графиков
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TemperatureDataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        
    def preprocess(self):
        # Переименование столбцов
        self.df = self.df.rename(columns={
            'Entity': 'Country',
            'year': 'Year',
            'Average surface temperature': 'Daily average temp',
            'Average surface temperature.1': 'Yearly average temp'
        })
        self.df = self.df.drop(columns=['Code'])
        
        # Преобразование даты
        self.df['Day'] = pd.to_datetime(self.df['Day'])
        self.df['Month'] = self.df['Day'].dt.month
        
        # Целевая переменная
        self.df['Target'] = self.df.groupby('Country')['Daily average temp'].shift(-1)
        self.df = self.df.dropna()
        self.df = self.df.sort_values(by=['Country', 'Year'])
    
    def get_country_data(self, country_name):
        country_data = self.df[self.df['Country'] == country_name].copy()
        country_data['Date'] = pd.to_datetime(country_data[['Year', 'Month']].assign(DAY=1))
        country_data = country_data.set_index('Date').sort_index()
        return country_data


class TemperatureForecaster:
    def __init__(self, data):
        self.data = data

    def sarima_forecast(self, steps=24, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        # Обучение SARIMA модели
        model = SARIMAX(self.data['Yearly average temp'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        
        # Прогноз
        forecast = results.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        return forecast_values, confidence_intervals

    def ets_forecast(self, steps=24):
        # Обучение ETS модели
        model = ExponentialSmoothing(self.data['Yearly average temp'], trend="add", seasonal="add", seasonal_periods=12)
        results = model.fit()
        
        # Прогноз
        forecast_values = results.forecast(steps=steps)
        return forecast_values

    def visualize_forecast(self, actual, forecast, confidence_intervals=None, title="Прогноз", file_name="forecast.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Фактические данные")
        plt.plot(forecast.index, forecast, label="Прогноз", linestyle="--", color="orange")
        
        if confidence_intervals is not None:
            plt.fill_between(forecast.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], 
                             color='orange', alpha=0.2, label="Доверительный интервал")
        
        plt.title(title)
        plt.xlabel("Дата")
        plt.ylabel("Средняя температура")
        plt.legend()
        plt.grid(True)
        
        # Сохранение графика
        output_path = os.path.join(OUTPUT_DIR, file_name)
        plt.savefig(output_path)
        logging.info(f"График '{title}' сохранён в файл: {output_path}")
        plt.show()


class LSTMForecaster:
    def __init__(self, data, look_back=12):
        self.data = data
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.model = Sequential()

    def prepare_data(self):
        scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:i + self.look_back])
            y.append(scaled_data[i + self.look_back])
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
    
    def build_model(self):
        self.model.add(LSTM(50, activation='relu', input_shape=(self.look_back, 1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, epochs=80, batch_size=16):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def forecast(self, steps=24):
        predictions = []
        input_sequence = self.X_test[-1]
        
        for _ in range(steps):
            input_sequence = input_sequence.reshape((1, self.look_back, 1))
            pred = self.model.predict(input_sequence)
            predictions.append(pred[0, 0])
            pred = pred.reshape(1, 1, 1)  # Добавляем reshape
            input_sequence = np.append(input_sequence[:, 1:, :], pred, axis=1)
        
        # Восстановление масштабов
        forecast_values = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Создание временного индекса для прогноза (после последней известной даты)
        start_date = self.data.index[-1]  # Последняя известная дата
        forecast_index = pd.date_range(start=start_date, periods=steps + 1, freq="MS")[1:]  # Пропускаем стартовую дату

        return forecast_values, forecast_index

    def visualize_forecast(self, actual, forecast, forecast_index, title="Прогноз LSTM", file_name="lstm_forecast.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Фактические данные")
        plt.plot(forecast_index, forecast, label="Прогноз", linestyle="--", color="orange")
        plt.title(title)
        plt.xlabel("Дата")
        plt.ylabel("Средняя температура")
        plt.legend()
        plt.grid(True)
        
        # Сохранение графика
        output_path = os.path.join(OUTPUT_DIR, file_name)
        plt.savefig(output_path)
        logging.info(f"График '{title}' сохранён в файл: {output_path}")
        plt.show()