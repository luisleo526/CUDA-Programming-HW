CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_86 -code=sm_86 -Xptxas -v  --compiler-options -fopenmp

BIN     := hist
LIB     := 

all: hist

hist: hist.cu
	$(NVCC) -o hist $(NVFLAGS) hist.cu -I$(LIB)
clean:
	rm -f $(BIN)

