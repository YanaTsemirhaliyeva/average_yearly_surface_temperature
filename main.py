import matplotlib.pyplot as plt
import pandas as pd
import logging

from src.data_loader import DataLoader
from src.data_info import DataInfo
from src.temperature_forecast import TemperatureDataProcessor, TemperatureForecaster, LSTMForecaster
from src.model_analysis import ModelAnalysis

# Путь к данным
file_path = "data/average-monthly-surface-temperature.csv"


if __name__ == "__main__":
    try:
        # Загрузка данных 
        loader = DataLoader()
        loader.load_csv(file_path)
        loader.data.rename(columns={
            'Entity': 'Country',
            'year': 'Year',
            'Average surface temperature': 'Daily average temp',
            'Average surface temperature.1': 'Yearly average temp'
            }, inplace=True)
        
        # Анализ данных
        loader.count_missing_values()
        data_info = DataInfo(loader.data)
        data_info.statistical_summary()

        # прогнозирование
        logging.info("=== Загрузка и предобработка данных ===")
        processor = TemperatureDataProcessor(file_path)
        processor.preprocess()
        country_data = processor.get_country_data("Belarus")
        
        # Прогноз с помощью SARIMA
        logging.info("=== Прогноз с помощью SARIMA ===")
        forecaster = TemperatureForecaster(country_data)
        forecast_values, confidence_intervals = forecaster.sarima_forecast(steps=24)
        forecaster.visualize_forecast(
            country_data['Yearly average temp'], 
            forecast_values, 
            confidence_intervals, 
            title="Прогноз SARIMA для Беларуси", 
            file_name="sarima_forecast_belarus.png"
        )
        
        # Прогноз с помощью ETS
        logging.info("=== Прогноз с помощью ETS ===")
        forecast_values_ets = forecaster.ets_forecast(steps=24)
        forecaster.visualize_forecast(
            country_data['Yearly average temp'], 
            forecast_values_ets, 
            title="Прогноз ETS для Беларуси", 
            file_name="ets_forecast_belarus.png"
        )

        # Прогноз с помощью LSTM
        logging.info("=== Прогноз с помощью LSTM ===")
        lstm_forecaster = LSTMForecaster(country_data['Yearly average temp'])
        lstm_forecaster.prepare_data()
        lstm_forecaster.build_model()
        lstm_forecaster.train(epochs=80)
        forecast_values_lstm, forecast_index_lstm = lstm_forecaster.forecast(steps=24)

        # Визуализация LSTM прогноза
        lstm_forecaster.visualize_forecast(
            actual=country_data['Yearly average temp'],
            forecast=forecast_values_lstm,
            forecast_index=forecast_index_lstm,
            title="Прогноз LSTM для Беларуси",
            file_name="lstm_forecast_belarus.png"
        )
        # Обучение моделей и оценка
        logging.info("=== Обучение моделей и оценка ===")
        analyzer_models = ModelAnalysis(file_path)
        analyzer_models.load_and_preprocess_data()
        analyzer_models.train_and_evaluate_models()

        logging.info("Программа успешно завершена!")

    except Exception as e:
        logging.error("Критическая ошибка в основном потоке: %s", e)