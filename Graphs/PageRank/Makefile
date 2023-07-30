all: default 

default: pageRank.cpp main.cpp
	g++ -I ../ -std=c++11 -fopenmp -g -O3 -o pr main.cpp pageRank.cpp ../Common/graph.cpp 
clean:
	rm -rf pr *~ *.*~
