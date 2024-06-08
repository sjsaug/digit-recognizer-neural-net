from python import Python

fn use_array() raises:
   # This is equivalent to Python's `import numpy as np`
   var np = Python.import_module("numpy")

   # Now use numpy as if writing in Python
   var array = np.array([1, 2, 3])
   print(array)

def main():
    use_array()