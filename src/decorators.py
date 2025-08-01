import functools
import datetime
import logging
import time

_logger = logging.getLogger()

def timer(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        start_time = time.perf_counter()
        
        _logger.info(f'{datetime.datetime.now()}: Executing function `{func.__name__}` ...')
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        run_time = end_time - start_time 
        _logger.info(f'{datetime.datetime.now()}: Finish function `{func.__name__}` in {run_time:.3f} secs.')
        
        return value
        
    return wrapper_decorator


def error_log(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            value = func 
        except Exception as e:
            _logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
        return value
    
    return wrapper_decorator