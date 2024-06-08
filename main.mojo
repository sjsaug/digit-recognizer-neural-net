from python import Python

fn use_array() raises:
   # This is equivalent to Python's `import numpy as np`
   var np = Python.import_module("numpy")
   var pd = Python.import_module("pandas")
   var mpl = Python.import_module("matplotlib")

   var data = pd.read_csv("data/train.csv")
   data.head()
   # Extract data
   data = np.array(data)
   var m = data.shape[0]
   var n = data.shape[1]
   np.random.shuffle(data)
   
   # Create a Python function that slices the array
   var slice_array = Python.compile_function("def slice_array(data, start, end): return data[start:end]")
   var data_dev = slice_array(data, 0, 1000).T

def main():
    use_array()