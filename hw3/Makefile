CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -Xptxas -v 

BIN     := poisson3d
LIB     :=

all: $(BIN)

$(BIN): poisson3d.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) poisson3d.cu $(LIB)

clean:
	rm -f $(BIN)

