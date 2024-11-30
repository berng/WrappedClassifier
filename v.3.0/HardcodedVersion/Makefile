CC = gcc
all: classifierV3.o test.c
	$(CC) test.c classifierV3.o -lm -o test
classifierV3.o: classifierV3.c classifierV3.h classifierV3_coefs0.h classifierV3_coefs1.h classifierV3_coefs2.h
	$(CC) classifierV3.c -lm -c -o classifierV3.o
clean:
	rm -f test *.o
