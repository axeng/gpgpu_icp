# gpgpu_icp

## 1 - Project Requirements

### 1.1 - Git submodules
This project is using git submodules, you thus need to clone the submodule in addition to the main project.
To do so run the following command:
```shell script
$ git submodule init
$ git submodule update
```

### 1.2 - CMake build system
In order to build this project you need to install the cmake build tool.

## 2 - Build the Project

### 2.1 - Build
Run the following command to build the project using cmake
```shell script
$ mkdir build
$ cd build
$ cmake ..
```

### 2.2 - Compile
Then you need to compile the programs
```shell script
$ cd build
$ make all
```

This will create four binaries `cpu`, `gpu_1`, `gpu_2`, `gpu_3` and `gpu_final`.

## 3 - Run the Project
All of the four binaries take as the first argument the destination file containing the cloud point as a csv and as a second argument the source
file containing the cloud point as a csv.

Here is an exemple:
```shell script
$ cd build
$ ./cpu destination.csv source.csv
```

## 4 - Run the Benchmark
To run the benchmark you first need to compile it using this command:
```shell script
$ cd build
$ make bench
```

Then you can execute it this way:
```shell script
$ cd build
$ ./bench
```

You can use the -h option to see the list of arguments the bench binary takes.
