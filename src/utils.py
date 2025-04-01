import logging
import os

def setup_logging():
    """Настраивает логирование."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'app.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_message(message):
    """Логирует сообщение."""
    logging.info(message)