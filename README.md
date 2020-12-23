# ActiveCrystal_CUDA
This CUDA code is used to generate configurations of Active Crystal systems, where the active particles form a triangular crystal and are connected permanently by springs.

To run this file, remember to create a new folder named "dataFiles" in the same folder. Then use the command: 

```c++
nvcc -std=c++11 kernal.cu -o test
```

to generate an excetable file.