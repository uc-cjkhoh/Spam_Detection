import os
import logging
import datetime

from . config_loader import cfg

today = datetime.datetime.now().strftime('%Y%m%d')
log_path = f'{cfg.module_log.path}/spam_detection_{today}.log'

# make sure the log folder exists
os.makedirs(cfg.module_log.path, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s'
)