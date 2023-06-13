CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_86 -code=sm_86 -Xptxas -v  --compiler-options -fopenmp

BIN     := monte_carlo
LIB     := 

all: monte_carlo

monte_carlo: monte_carlo.cu
	$(NVCC) -o monte_carlo $(NVFLAGS) monte_carlo.cu
clean:
	rm -f $(BIN)

