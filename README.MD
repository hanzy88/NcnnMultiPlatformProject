# Sample project to use NCNN for multi-platform
## How to try
### Common (Getting source code)
```sh
git clone https://github.com/take-iwiw/NcnnMultiPlatformProject.git
cd NcnnMultiPlatformProject
git submodule init
git submodule update
```

### Windows (Visual Studio 2017)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
- Start `MyNcnnProject.sln`
- Set `MyProject` project as a startup project, then build and run!

This project uses pre-built ncnn library by default.
If you want to create ncnn library by yourself or work on another version of Visual Studio, please disable `USE_PREBUILT_NCNN` option in cmake-gui

### PC Linux (Ubuntu)
```sh
cd NcnnMultiPlatformProject
mkdir build && cd build
cmake ..
make
cd MyProject && ./MyProject
```

### Raspberry Pi 3 (Native build)
```sh
cd NcnnMultiPlatformProject
mkdir build && cd build
cmake .. -DBUILD_TARGET=RASPI_NATIVE
make
cd MyProject && ./MyProject
```

### Raspberry Pi 3 (Cross build on Ubuntu)
```sh
cd NcnnMultiPlatformProject
mkdir build && cd build
cmake .. -DBUILD_TARGET=RASPI_CROSS
make
cd MyProject && ./MyProject
```


