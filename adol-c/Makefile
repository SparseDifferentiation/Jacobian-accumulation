ADOLC_PREFIX = $(CURDIR)/ADOL-C/local
CXXFLAGS = -std=c++20 -I$(ADOLC_PREFIX)/include
LDFLAGS = -L$(ADOLC_PREFIX)/lib -ladolc -Wl,-rpath,$(ADOLC_PREFIX)/lib

sparse_jacobian: sparse_jacobian.cpp
	g++ $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f sparse_jacobian

.PHONY: clean
