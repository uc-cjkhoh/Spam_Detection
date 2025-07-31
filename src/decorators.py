import functools
import datetime
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        start_time = time.perf_counter()
        
        print(f'[INFO] {datetime.datetime.now()}: Executing function `{func.__name__}` ...')
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        run_time = end_time - start_time 
        print(f'[INFO] {datetime.datetime.now()}: Finish function `{func.__name__}` in {run_time:.3f} secs.')
        
        return value
        
    return wrapper_decorator