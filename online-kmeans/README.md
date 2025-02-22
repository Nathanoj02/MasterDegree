# Signals
Project for Signals, Image &amp; Video, Master's Degree in Artificial Intelligence Systems

## Setup
Clone repository
```
git clone git@github.com:Nathanoj02/online-kmeans.git
```

Update submodule
```
git submodule update --init --recursive
```

Cpp setup
```
cd cpp && mkdir build && cd build
```
```
cmake ..
```
```
make -j
```

## Execution
Python execution
```
cd lib
```
```
python3 main.py
```

## Arguments
Operation (image, video)
```
-o image
```

Source image / video path
```
-s path
```

Destination image / video path
```
-d path
```

### Optional arguments
k value (int)
```
-k value
```

Max tolerance / error value (float)
```
-e value
```

Algorithm (only for image) (python, cpp, cuda, cudashared)
```
-a algorithm
```

## Utils
Recompile C++ / CUDA from lib folder
```
cd ../cpp/build && cmake .. && make -j8 && cd ../../lib
```
