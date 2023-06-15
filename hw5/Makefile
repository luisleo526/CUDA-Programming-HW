CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_86 -code=sm_86 -Xptxas -v  --compiler-options -fopenmp

BIN     := laplace2d_NGPU
LIB     :=

all: laplace2d_NGPU

laplace2d_NGPU: laplace2d_NGPU.cu
	$(NVCC) -o laplace2d_NGPU $(NVFLAGS) laplace2d_NGPU.cu $(LIB)
clean:
	rm -f $(BIN)

