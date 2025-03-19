# Kaldi setup alternatives

We illustrate two tested alternatives for setting up a working kaldi environment.
1. [Kaldi Image](https://hub.docker.com/r/kaldiasr/kaldi)
2. Manual (Latest) Instructions

## 1. Docker Image
```bash
docker pull kaldiasr/kaldi
# cpu-based image
docker run -it kaldiasr/kaldi:latest
```

> [!WARNING]
> Mount your code and data as volumes. If you don’t, any files created inside the container will be lost when it stops.

> [!TIP]
> You can use `bind mounts`, `docker volumes`, `docker compose` or other solutions.


## 2. Manual installation (ubuntu 22.04, python 3)

> Thanks a lot to the student who provided us with this updated version!

-
  ```bash
  sudo apt update
  sudo apt install gcc-9 g++-9 gfortran-9 git
  ```
- 
  ```bash
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
  sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 90
  ```
- 
  ```bash
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
  sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 90
  ```
- 
  ```bash
  git clone https://github.com/kaldi-asr/kaldi.git
  cd kaldi
  ```
-
  ```bash
  sudo apt install -y zip python3 python2.7 make automake autoconf sox libtool subversion gawk moreutils zlib1g-dev
  ```
- 
  ```bash
  cd tools
  extras/check_dependencies.sh
  ```
> [!Tip]
> In case of missing packages please install them as suggested - except from libatlas3-base, e.g., 
    ```extras/install_mkl.sh```

-
  ```bash
  rm -rf python/*
  mkdir -p python
  touch python/.use_default_python
  ```

- 
  ```bash
  extras/check_dependencies.sh
  ```
- 
  ```bash
  # The number of jobs can be modified. Can also use -j2 for smaller machines or -j8 for larger ones
  make -j4
  ```
- 
  ```bash
  extras/install_irstlm.sh
  ```
- 
  ```bash
  extras/install_openblas.sh
  ```
> [!Tip]
> In case of error with `TARGET` variable set it manually. 1) Find your cpu type via ```cat /proc/cpuinfo | grep "model name" | head -1``` or ```lscpu | grep "Model name"```. 2) Depending on the cpu type, e.g., `HASWELL` (intel), `SKYLAKEX` (intel), `ZEN` (amd ryzen), `CORE2` (generic x_86_64), set `TARGET` manually by opening the file as ```nano extras/install_openblas.sh``` and in the line that starts with `make` modify as `make PREFIX=$(pwd)/OpenBLAS/install TARGET=CORE2 USE_LOCKING=1 USE_THREAD=0 -C OpenBLAS all install` and save the changes. 3) rerun ```extras/install_openblas.sh```

-
  ```bash
  cd ../src
  ./configure --shared --openblas-root=../tools/OpenBLAS/install
  ```
- 
  ```bash
  make clean -j4
  make depend -j4
  make -j4
  ```


- Final check step 
  ```bash
  cd kaldi/egs/yesno/s5
  ./run.sh
  ```

- Expected Output
  ```bash
  local/score.sh: scoring with word insertion penalty=0.0,0.5,1.0
  %WER 0.00 [ 0 / 232, 0 ins, 0 del, 0 sub ] exp/mono0a/decode_test_yesno/
  wer_10_0.0
  ```

