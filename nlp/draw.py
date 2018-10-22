import numpy as np
import matplotlib.pyplot as plt
import informative_terms from main

def draw_embedding_matrix(embedding_matrix):
    for term_index in range(len(informative_terms)):
      # Create a one-hot encoding for our term.  It has 0s everywhere, except for
      # a single 1 in the coordinate that corresponds to that term.
      term_vector = np.zeros(len(informative_terms))
      term_vector[term_index] = 1
      # We'll now project that one-hot vector into the embedding space.
      embedding_xy = np.matmul(term_vector, embedding_matrix)
      plt.text(embedding_xy[0],
               embedding_xy[1],
               informative_terms[term_index])

    # Do a little setup to make sure the plot displays nicely.
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    plt.show() 