CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -Xptxas -v --compiler-options -fopenmp

BIN     := hist
LIB     := 

all: hist

hist: hist.cu
	$(NVCC) -o hist $(NVFLAGS) hist.cu $(LIB)
clean:
	rm -f $(BIN)

