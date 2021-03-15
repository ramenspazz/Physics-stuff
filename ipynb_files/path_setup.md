# To include the python modules in py_files
> Add this line to be begining of your ipynb file
```python
import path_setup
```
# Adding a Data directory
> Add these lines to your code:
```python
nb_dir = path_setup.path_setup()
data_dir = nb_dir + '/Data/'
```
