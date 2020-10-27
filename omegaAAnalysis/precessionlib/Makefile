rootlibs = $(shell root-config --glibs)
rootflags = $(shell root-config --cflags)
warnings = -Wall -Wextra

all: hist3dToNumpyArray.so

hist3dToNumpyArray.so: hist3dToNumpyArray.cxx
	g++ $^ -fPIC -shared -O2 -o $@  $(warnings) $(rootflags) $(rootlibs)
