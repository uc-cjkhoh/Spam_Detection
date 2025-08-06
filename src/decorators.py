import functools
import datetime
import logging
import time
import os 

from . config_loader import cfg

today = datetime.datetime.now().strftime('%Y%m%d')
log_path = f'{cfg.log.path}/spam_detection_{today}.log'

os.makedirs(cfg.log.path, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s'
)

def timer(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        start_time = time.perf_counter()
        
        logging.info(f'Executing function `{func.__name__}` ...')
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        run_time = end_time - start_time 
        logging.info(f'Finish function `{func.__name__}` in {run_time:.3f} secs.')
        
        return value
        
    return wrapper_decorator


def error_log(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
        return value
    
    return wrapper_decorator