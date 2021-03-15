import os
import sys
def path_setup():
    #import external modules
    nb_dir = os.path.split(os.getcwd())[0]
    parent_dir = nb_dir + '/Py_files'
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    return nb_dir