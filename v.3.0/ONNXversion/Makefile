CC = gcc
CPP = g++
ONNXLIB = onnxruntime-linux-x64-1.20.1
CFLAGS_ONNX = -I$(ONNXLIB)/include
LDFLAGS_ONNX =  -L$(ONNXLIB)/lib  -lonnxruntime 

all: classifier_func.o test.c
	$(CPP) test.c classifier_func.o $(CFLAGS_ONNX) $(LDFLAGS_ONNX) -lm -o test

clean:
	rm -f test *.o

classifier_func.o: classifier_func.cpp classifier_func.h
	$(CPP) classifier_func.cpp $(CFLAGS_ONNX) $(LDFLAGS_ONNX) -c -o classifier_func.o
