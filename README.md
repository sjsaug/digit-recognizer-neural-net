## Digit Recognizer Neural Net

### About
Neural Network made using ONLY Linear Algebra + Numpy.
<br/>Uses the Digit Recognizer dataset (https://www.kaggle.com/competitions/digit-recognizer).
<br/>Trains a Neural Network to recognize digits.
<br/>To view the Linear Algebra behind this project, check `NN Math.jpg`

### Usage
1. Make sure the following dependancies are installed with `pip install` :
    - numpy
    - pandas
    - matplotlib
2. Specify your desired training iterations and alpha (default is **(5000, 0.001)**).
    - *NOTE : Recommended to only change iterations. Higher accuracy will require more iterations exponentially. <br/>With 1k iterations we get 74% accuracy | 5k iterations we get 87% accuracy | 10k iterations we get 89% accuracy <br/>For 99%+ accuracy we would likely need anywhere from 300k-1M+ iterations*
3. To start training, run `python3 main.py` in your project directory.
4. Done! Project will auto test after training is complete.

### Benchmarks
System 1 (Laptop) : M2 Pro MBP (Base Config 16GB RAM)
- System 1 gets around 10 iterations per second, slowing down to 10 iterations per 1.5 seconds after the first 100 iterations. Noticeable tempeature increase as iterations increase which explains the performance decrease.

<br/>System 2 (SFF Desktop) : Watercooled 5700X3D + 3090 + 64GB DDR4 3600
- System 2 gets around 10 iterations per 1.2 seconds, not slowing down (measured up until 2.5k iterations). Temperature remained the same which explains the consistency - would be interesting to test this with a 5950X or Threadripper.

### Contribution Opportunities
Here are just a few things that can be contributed to improve the project :
1. Configuration Support : Add config system that can retrieve data such as iteration amount & alpha from a desired config file, loaded on start.
2. Separate Training & Testing : Add option to choose between training the model and testing it.
3. Testing with different systems. The main performance determinants are the CPU and RAM of the system, as that is what Numpy is limited to by Python.
