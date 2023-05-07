# cppraisr
A C++ implementation of [RAISR](https://ieeexplore.ieee.org/document/7744595).

# Prerequisites

- [CMake 3.18<=](https://cmake.org/)

# Building

## Windows

```bat
$ mkdir build & cd build
$ cmake .. -G"Visual Studio 17 2022"
```

## Linux

```shell
$ mkdir build && cd build
$ cmake .. && make
```

# Training
Put images for traing in the train_data directory.

```shell
$ ./bin/train
```

You can use some options.
```
usage: train [-help] [-f Filter] [-max Max Images] [-threads Number of Threads]
arguments:
        -help   show this help
        -f      a filter name to write without extension
        -max    the max number of training images
        -threads        the number of threads to use
```


# Testing
`./bin/test` tests images under the test_data.

```
usage: test [-help] [-f Filter] [-max Max Images] [-q]
arguments:
        -help   show this help
        -f      a filter to use
        -max    the max number of testing images
        -q      switch to quality testing mode, measure the avarage of each MSSIMs.
```

# References
- Y. Romano, J. Isidoro and P. Milanfar, "RAISR: Rapid and Accurate Image Super Resolution" in IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 110-125, March 2017.

# License
This software is distributed under two licenses, MIT License or Public Domain, choose whichever you like.