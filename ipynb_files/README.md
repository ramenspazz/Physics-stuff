# Using Jupyter notebooks with these modules
If you are using a Jupyter notebook (ipynb extenstion), you can place them in the ipynb_files directory for easy project management and organization.
# To include the python modules in a Jupyter notebook
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
