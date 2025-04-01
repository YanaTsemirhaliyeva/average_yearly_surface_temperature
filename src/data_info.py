import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Логирование в файл
        logging.StreamHandler()         # Логирование в консоль
    ]
)

class DataInfo:
    def __init__(self, data: pd.DataFrame):
        """
        Инициализирует анализатор датасета.

        :param data: DataFrame для анализа.
        """
        self.data = data

    def dataset_head_info(self) -> None:
        try:
            logging.info(f"Размер датасета: {self.data.shape}")
            logging.info(f"Типы данных в каждом столбце:\n{self.data.dtypes}")
        except Exception as e:
            logging.error(e, "Ошибка при получении информации о датасете")

    def statistical_summary(self) -> None:
        """
        Выводит основные статистики для числовых столбцов.
        """
        try:
            stats = self.data.describe()
            logging.info(f"Основные статистики для числовых столбцов:\n{stats}")
        except Exception as e:
            logging.error(e, "Ошибка при вычислении основных статистик")

    def correlation_analysis(self) -> None:
        """
        Выводит матрицу корреляций и строит тепловую карту.
        """
        try:
            correlation_matrix = self.data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Матрица корреляций")
            plt.show()
        except Exception as e:
            logging.error(e, "Ошибка при анализе корреляции")

    def detect_outliers(self, threshold: float = 3.0) -> None:
        """
        Поиск выбросов с использованием Z-оценки.

        :param threshold: Порог Z-оценки для определения выбросов.
        """
        try:
            numeric_data = self.data.select_dtypes(include=["float64", "int64"])
            z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()
            outliers = (abs(z_scores) > threshold).sum()
            logging.info(f"Количество выбросов в каждом столбце:\n{outliers}")
        except Exception as e:
            logging.error(e, "Ошибка при обнаружении выбросов")