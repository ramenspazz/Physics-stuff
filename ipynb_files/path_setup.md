# To include the python modules in py_files
> Add this line to be begining of your ipynb file
> cat('\x60', '\n')
`python '\x60import path_setup\x60'`
# Adding a Data directory
> Add these lines to your code:
`python '\x60nb_dir = path_setup.path_setup()\x60'`
`python '\x60data_dir = nb_dir + '/Data/'\x60'`