# libcapsule — Context Capsules prototype (package 1)
# Targets: all (dylib + test_capsule + bench_micro + sec_c), test, clean.
# CommonCrypto lives in libSystem: no extra link flags needed. -lz allowed
# by the design but not required by this implementation.

CC      = clang
CFLAGS  = -O2 -std=c11 -Wall -Wextra -fPIC
LIBOBJS = capsule.o cap.o hash.o evict.o
HDRS    = capsule.h store_internal.h bench_util.h brlock.h

all: libcapsule.dylib test_capsule bench_micro sec_c bench_conc

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

libcapsule.dylib: $(LIBOBJS)
	$(CC) -dynamiclib -o $@ $(LIBOBJS)

test_capsule: test_capsule.o $(LIBOBJS)
	$(CC) $(CFLAGS) -o $@ test_capsule.o $(LIBOBJS)

bench_micro: bench_micro.o $(LIBOBJS)
	$(CC) $(CFLAGS) -o $@ bench_micro.o $(LIBOBJS)

sec_c: sec_c.o $(LIBOBJS)
	$(CC) $(CFLAGS) -o $@ sec_c.o $(LIBOBJS)

bench_conc: bench_conc.o $(LIBOBJS)
	$(CC) $(CFLAGS) -o $@ bench_conc.o $(LIBOBJS) -lpthread

test: test_capsule
	CAPSULE_TEST_ROOT=$(CURDIR)/../../experiments/tmp_store_c ./test_capsule

clean:
	rm -f *.o libcapsule.dylib test_capsule bench_micro sec_c bench_conc

.PHONY: all test clean
