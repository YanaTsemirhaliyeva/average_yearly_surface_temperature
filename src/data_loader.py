import os
import pandas as pd
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


class DataLoader:
    def __init__(self):
        self.data = None

    def load_csv(self, path: str) -> None:
        """
        Загружает данные из CSV-файла.

        :param path: Путь к файлу CSV.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл {path} не существует.")
            self.data = pd.read_csv(path)
            logging.info(f'Данные успешно загружены из файла: {path}')
        except Exception as e:
            logging.error(e, f'Ошибка при загрузке CSV из файла {path}')

    def watch_data_head(self, n: int = 5) -> None:
        """
        Выводит первые строки данных.

        :param n: Количество строк для отображения.
        """
        try:
            if self.data is not None:
                logging.info(f'Первые {n} строк данных:\n{self.data.head(n)}')
            else:
                logging.info('Данные не загружены. Используйте метод load_csv().')
        except Exception as e:
            logging.error(e, "Ошибка при выводе первых строк данных")

    def count_missing_values(self) -> None:
        """
        Выводит количество пропущенных значений в каждом столбце.
        """
        try:
            if self.data is not None:
                missing_values = self.data.isnull().sum()
                logging.info('Количество пропущенных значений в каждом столбце:')
                logging.info(f'\n{missing_values}')
            else:
                logging.error('Данные не загружены. Используйте метод load_csv().')
        except Exception as e:
            logging.error(e, "Ошибка при подсчёте пропущенных значений")

    def fill_missing_values(self, strategy: str = 'mean') -> None:
        """
        Заполняет пропущенные значения в данных в соответствии с указанной стратегией.

        :param strategy: Стратегия заполнения ('mean', 'median', 'most_frequent').
        """
        if strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError("Неподдерживаемая стратегия заполнения. Используйте 'mean', 'median' или 'most_frequent'.")
       
        try:
            if self.data is not None:
                if self.data.isnull().sum().sum() == 0:
                    logging.info("Нет пропущенных значений. Заполнение не требуется.")
                    return

                if strategy == 'mean':
                    self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
                elif strategy == 'median':
                    self.data.fillna(self.data.median(numeric_only=True), inplace=True)
                elif strategy == 'most_frequent':
                    self.data.fillna(self.data.mode().iloc[0], inplace=True)
                else:
                    logging.info("Неподдерживаемая стратегия заполнения. Используйте 'mean', 'median' или 'most_frequent'.")
                    return

                logging.info(f"Пропущенные значения успешно заполнены с использованием стратегии: {strategy}.")
            else:
                logging.info("Данные не загружены. Используйте метод load_csv().")
        except Exception as e:
            logging.error(e, "Ошибка при заполнении пропущенных значений")