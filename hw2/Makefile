CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g 
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -Xptxas -v 

BIN     := vecRedu
LIB     :=

all: $(BIN)

$(BIN): vecRedu.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) vecRedu.cu $(LIB)

clean:
	rm -f $(BIN)

