#include <Python.h>

static PyObject* total_least_squares(PyObject* self, PyObject* args){
    char *msg = "total_least_squares";
    return Py_BuildValue("s", msg)
}