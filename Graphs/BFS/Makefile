all: default 

default: main.cpp bfs.cpp
	g++ -I ../ -std=c++11 -fopenmp -O3 -g -o bfs main.cpp bfs.cpp ../Common/graph.cpp 
clean:
	rm -rf bfs *~ *.*~
