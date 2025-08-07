import os
import logging
import datetime

from . config_loader import cfg

os.makedirs(cfg.module_log.general_log_path.folder, exist_ok=True)
os.makedirs(cfg.module_log.process_log_path.folder, exist_ok=True)

today = datetime.datetime.now().strftime('%Y%m%d')
log_path = f'{cfg.module_log.general_log_path.folder}/spam_detection_{today}.log'

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s'
)