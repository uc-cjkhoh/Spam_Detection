import pandas as pd
import numpy as np

from prefect import task, flow, get_run_logger
from prefect.cache_policies import NO_CACHE

from src.utils.util import get_unique_pattern_ids


@flow(name='Setup Initial Data')
def main():
    logger = get_run_logger()
    
    pass



if __name__ == '__main__':
    main()
