import functools 
import time 

from loader.logger_loader import logging 
  
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