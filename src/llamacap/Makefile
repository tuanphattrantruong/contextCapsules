# llamacap — capsules mounted under llama.cpp's per-sequence KV state API.
#
# Point LLAMA_DIR at a built llama.cpp tree (see build_llama.sh):
#   make LLAMA_DIR=/tmp/llama.cpp

LLAMA_DIR ?= /tmp/llama.cpp
LLAMA_INC  = $(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LLAMA_LIB  = $(LLAMA_DIR)/build/bin

CC      = clang
CFLAGS  = -O2 -std=c11 -Wall -Wextra -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I../libcapsule
LDFLAGS = -L$(LLAMA_LIB) -lllama -Wl,-rpath,$(LLAMA_LIB)
CAPOBJS = ../libcapsule/capsule.o ../libcapsule/cap.o \
          ../libcapsule/hash.o ../libcapsule/evict.o

all: llamacap

$(CAPOBJS):
	$(MAKE) -C ../libcapsule

llamacap: llamacap.c $(CAPOBJS)
	$(CC) $(CFLAGS) -o $@ llamacap.c $(CAPOBJS) $(LDFLAGS)

clean:
	rm -f llamacap

.PHONY: all clean
