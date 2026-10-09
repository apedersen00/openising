CC       = gcc
CFLAGS   = -Wall -ggdb
# rudy.c and gb_lib.c are pre-ANSI (K&R) C
KRFLAGS  = -std=gnu89 -w -Wno-error=implicit-function-declaration

all: rudy sdp

rudy: gb_lib.c rudy.c
	${CC} $(KRFLAGS) $^ -o rudy -lm

sdp: g2sdp.c
	${CC} $(CFLAGS) $< -o g2sdp -lm

clean:
	rm -f rudy g2sdp *.o

.PHONY: all sdp clean
