#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  /*
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  */

  /*
     TODO STUDENTS: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

  //
  int numNodes = num_nodes(g);
  double *score_new = (double*)calloc(numNodes, sizeof(double));
  double *score_old = (double*)calloc(numNodes, sizeof(double));
  double equal_prob = 1.0 / numNodes;
  double global_diff = convergence;
  double out_val = 0.0;
  
  int size = 0;
  int *indexes = (int*)calloc(numNodes, sizeof(int));

  for (int i = 0; i < numNodes; ++i) {
    score_old[i] = equal_prob;
    if (outgoing_size(g, i) == 0) {
      indexes[size] = i;
      size++;
    }
  }

  while ( !(global_diff < convergence) ) {
    global_diff = 0.0;
    out_val = 0.0;

    for (int i = 0; i < size; ++i) {
      out_val += score_old[indexes[i]];
    }

    #pragma omp parallel for reduction(+:global_diff) schedule(dynamic, 32)
    for (int i = 0; i < numNodes; ++i) {
      score_new[i] = 0.0;

      for (const Vertex *j = incoming_begin(g, i); j < incoming_end(g, i) ; ++j) {
        score_new[i] += score_old[*j]/outgoing_size(g, *j);
      }

      score_new[i] = (damping * score_new[i]) + (1.0 - damping) / numNodes;
      score_new[i] += damping * out_val / numNodes;

      global_diff += fabs(score_new[i] - score_old[i]);
    }

    std::swap(score_new, score_old);
  }

  memcpy(solution, score_old, sizeof(double)*numNodes);

  free(indexes);
  free(score_new);
  free(score_old);
}
