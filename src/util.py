import os
 
from . decorators import timer, error_log

@error_log
@timer
def is_folder_empty(folderpath: str) -> bool:
    """
    Check if folder is empty

    Args:
        folderpath (str): target folder's path

    Returns:
        bool: True if folder is empty else False
    """
    
    if not os.path.isdir(folderpath):
        print(f'Folder {folderpath} not exists, creating folder ...')
        return False
    
    return not os.listdir(folderpath)