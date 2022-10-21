#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define THRESHOLD 1e6

//#define VERBOSE

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    graph* g,
    vertex_set* frontier,    
    int* distances,    
    int depth)
{
    int cnt = 0;
    #pragma omp parallel 
    {
        #pragma omp for reduction(+:cnt) schedule(dynamic, 5096)
        for (int i = 0; i < g->num_nodes; ++i) {   
            if (frontier->vertices[i] == depth) {
                int start_edge = g->outgoing_starts[i];
                int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[i+1];

                // attempt to add all neighbors to the new frontier            
                for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
                    int outgoing = g->outgoing_edges[neighbor];
                    if(frontier->vertices[outgoing] == NOT_VISITED_MARKER) {                
                        distances[outgoing] = distances[i]+1;
                        cnt++;
                        frontier->vertices[outgoing] = depth+1;
                    }
                }
            }
        }
    }
    frontier->count = cnt;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);    

    int depth = 0;

    vertex_set* frontier = &list1;

    memset(frontier->vertices, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        // explore with xth layer of depth
        top_down_step(graph, frontier, sol->distances, depth);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        // increase the depth of exploration
        depth++;
    }
}


void bottom_up_step(
    graph* g,
    vertex_set* frontier,    
    int* distances,    
    int depth)
{
    int cnt = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:cnt) schedule(dynamic, 5096)
        for (int i = 0; i < g->num_nodes; ++i) {
            if (frontier->vertices[i] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->incoming_starts[i+1];

                for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
                    int incoming = g->incoming_edges[neighbor];
                    if (frontier->vertices[incoming] == depth) {
                        distances[i] = distances[incoming]+1;
                        cnt++;
                        frontier->vertices[i] = depth+1;
                        break;
                    }
                }
            }
        }
    }
    frontier->count = cnt;
}

void bfs_bottom_up(Graph graph, solution* sol)
{

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    int depth = 0;

    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bottom_up_step(graph, frontier, sol->distances, depth);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time-start_time);
#endif

        depth++;
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    int depth = 0;

    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        if (frontier->count > THRESHOLD) {
            bottom_up_step(graph, frontier, sol->distances, depth);
        } else {
            top_down_step(graph, frontier, sol->distances, depth);
        }
        

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time-start_time);
#endif

        depth++;
    }
}
