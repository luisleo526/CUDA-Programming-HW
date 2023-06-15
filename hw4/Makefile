CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -Xptxas -v --compiler-options -fopenmp

BIN     := vecDot_NGPU
LIB     :=

all: $(BIN)

$(BIN): vecDot_NGPU.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) vecDot_NGPU.cu $(LIB)

clean:
	rm -f $(BIN)

