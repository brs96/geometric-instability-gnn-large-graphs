# Adapted from https://github.com/SGDE2020/embedding_stability/blob/master/lib/tools/comparison.py

import os
import pickle
from functools import partial
from itertools import combinations
import multiprocessing as mp
from time import time

import numpy as np
import psutil
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.neighbors import BallTree
from tqdm import tqdm
from sklearn.preprocessing import normalize


class Comparison:

    def __init__(self, embeddings, num_nodes, file_prefix):
        """
        embeddings: list, list of strings that specify the embedding files
        """
        self.embeddings = embeddings
        self.pairs = self._combinations(embeddings)
        self.num_vertices = num_nodes

        # use file names without numbering to mark result files
        self.file_prefix = file_prefix

    def _combinations(self, emb_list):
        """ Computes all different comparison pairs of a list of embeddings """
        return [pair for pair in combinations(emb_list, 2)]

    def _analyse_jaccard(self, queries, nodes, k, pair):
        """This function is called in the multiprocessing of jaccard score"""
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])
        nodes = np.asarray(nodes)
        jaccard_score = {}
        for i in range(len(nodes)):
            jaccard_score[nodes[i]] = \
                len(np.intersect1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)], assume_unique=True)) / \
                len(np.union1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)]))
        return list(jaccard_score.values())

    def _analyse_second_cossim(self, queries, normed_embs, nodes, k, pair):
        """
        This function is called in the multiprocessing of the second order cosine similarity.
        """

        # Convert the indices of nearest neighbors back into numpy
        indices_0 = np.asarray(queries[pair[0]])
        indices_1 = np.asarray(queries[pair[1]])

        # Convert the embeddings and nodes back into numpy
        norm_emb_0 = np.asarray(normed_embs[pair[0]])
        norm_emb_1 = np.asarray(normed_embs[pair[1]])
        nodes = np.asarray(nodes)

        # Compute the second order cosine similarity
        pair_results = []
        for i in range(len(nodes)):
            # Build the set of nearest neighbors w.r.t. both embeddings
            # Use indices from 1 to k+1, because the first entry will always be the node itself
            neighbors_union = np.union1d(indices_0[i, 1:(k + 1)], indices_1[i, 1:(k + 1)])

            # Vectors of cosine similarity values of nearest neighbors. There was an error in the original source code, did not use norm_emb_1 for m1
            m0 = cos_sim(norm_emb_0[neighbors_union], norm_emb_0[nodes[i]].reshape(1, -1))
            m1 = cos_sim(norm_emb_1[neighbors_union], norm_emb_1[nodes[i]].reshape(1, -1))

            # Flatten output matrix
            assert m0.shape[1] == 1 and m1.shape[1] == 1, "m0 and m1 should only have a single variable in the second dimension"
            m0 = m0.flatten()
            m1 = m1.flatten()

            # Cosine similarity between similarity vectors
            pair_results.append(float(1-cosine(m0,m1)))
        return pair_results

    def k_nearest_neighbors(self, nodes=None, append=False, samples=100, k=10, jaccard=False, load=False,
                            kload_size=100, save=True, save_path=None, num_processes=4):
        """
        Computes the k nearest neighbors to some specified nodes with respect to cosine similarity. As an intermediate
        step, it computes the 100 nearest neighbors and saves them to file. Based on these neighbors, the k-nn overlaps
        are computed.
        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            samples: int, number of nodes that will be sampled
            k: int, number of neighbors that will be used in the comparison
            jaccard: bool, whether jaccard score or overlap will be used as similarity measure
            load: bool, whether a file of computed neighbors will be used
            kload_size: Size of k in knn file to load
            save: bool, whether the results should be saved in a text file
            save_path: str, path where the file will be saved
            num_processes: number of random processes in parallelization
        Returns:
            dict, "nodes": array of nodes, "overlaps": array of overlaps, columns are values per node; or array of
            jaccard scores
        """
        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, samples, append)

        # Load k nearest neighbors to save time? Else we will have to compute them first.
        if load:
            file_name = self.file_prefix + "_" + str(kload_size) + "nns.pickle"
            with open(os.path.join(save_path, file_name), "rb") as pickle_file:
                queries = pickle.load(pickle_file)
            assert list(queries.keys()) == self.embeddings, ("Keys of loaded queries do not match"
                                                             " with available embeddings")

        # Normalizing the embeddings to be able to use distance as a proxy for cosine similarity
        # BallTree from sklearn is used to compute the neighbors efficiently
        else:
            queries = self.nearest_neighbors_queries(nodes, k, save_path)
        # Store the naive overlaps for all pairs

        print(f"\n\n {queries.keys()} \n")

        # Use multiprocessing to speed up overlap computation.
        # Too much data is passed to the processes which makes it inefficient.
        # Possibly, it is faster to store the data as a file as an intermediate step.
        # only run if less than 100 neighbors are queried, as too large neighborhoods may cause memory issues when
        # distributing tasks
        if num_processes > 1 and k <= 100:
            with mp.Pool(num_processes) as p:
                # arguments passed in multiprocessing must be pickable
                p_queries = queries
                p_nodes = nodes.tolist()
                if jaccard:
                    multiprocess_func = partial(self._analyse_jaccard, p_queries, p_nodes, k)
                else:
                    multiprocess_func = partial(self._analyse_knn, p_queries, p_nodes, k)
                li_overlap = []
                for result in tqdm(p.imap(multiprocess_func, self.pairs), total=len(self.pairs)):
                  li_overlap.append(result)
        else:
            if jaccard:
                li_overlap = [self._analyse_jaccard(queries, nodes.tolist(), k, pair) for pair in self.pairs]
            else:
                li_overlap = [self._analyse_knn(queries, nodes.tolist(), k, pair) for pair in self.pairs]

        # Convert the result into numpy
        overlap = np.asarray(li_overlap)

        # Save the results
        if jaccard:
            nodes_suffix = "jaccard_nodes"
            scores_suffix = f"{k}nn_jaccard"
        else:
            nodes_suffix = "overlap_nodes"
            scores_suffix = f"{k}nn_overlap"
        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_{nodes_suffix}"), nodes)
            np.save(os.path.join(save_path, f"{self.file_prefix}_{scores_suffix}"), overlap)
        return {"nodes": nodes, "overlaps": overlap}

    def jaccard_similarity(self, nodes=None, append=False, samples=100, k=10, load=False, kload_size=100, save=True,
                           save_path=None, num_processes=4):
        """
        Alias for k_nearest_neighbors with jaccard=True.
        See k_nearest_neighbors for detailed documentation.
        """
        return self.k_nearest_neighbors(nodes=nodes, append=append, samples=samples, k=k, jaccard=True, load=load,
                                        kload_size=kload_size, save=save, save_path=save_path,
                                        num_processes=num_processes)


    def second_order_cosine_similarity(self, nodes=None, append=False, num_samples=1000, k=10, load=True, save=False,
                                       save_path=None, num_processes=4):
        """ Computes second order cosine similarity.
        Args:
            nodes: list, that specifies the nodes
            append: bool, if nodes are specified, whether additional nodes will be sampled and added to nodes
            num_samples: int, number of nodes that will be sampled
            k: int, number of neighbors that will be used in the comparison
            load: bool, whether to load nearest neighbors from file
            save: bool, whether the results should be saved
            save_path: str, path where the file will be saved
            num_processes: number of random processes in parallelization
        Returns:
            nodes: numpy array of used nodes
            results: numpy array of similarity values of size (number of embedding pairs, number of nodes)
        """

        # Handle the nodes input: Whether to sample nodes, use given nodes, or both.
        nodes = self._get_nodes(nodes, num_samples, append)

        # Load required data: nearest neighbors, embeddings
        normed_embs = {}

        if load:
            # Load nearest neighbors from file
            file_name = self.file_prefix + "_" + str(k)+ "nns.pickle"
            with open(os.path.join(save_path, file_name), "rb") as pickle_file:
                queries = pickle.load(pickle_file)
            assert list(queries.keys()) == self.embeddings, ("Keys of loaded queries do not match with "
                                                             "available embeddings")
            for emb in tqdm(self.embeddings, desc="Loading nearest neighbor files"):
                normed_embs[emb] = normalize(self.read_embedding(emb), norm='l2', copy=False)
        else:
            queries, normed_embs = self.nearest_neighbors_queries(nodes, k, save_path, return_embeddings=True)

        # Start computation of second order cosine similarity
        # arguments passed in multiprocessing must be pickable
        p_normed_embs = dict([(key, norm_emb.tolist()) for key, norm_emb in normed_embs.items()])
        p_nodes = nodes.tolist()
        # Avoid multiprocessing on Colab, not sure if it works locally though
        if num_processes > 1 and k <= 100:
            with mp.Pool(num_processes) as p:
                li_results = p.map(partial(self._analyse_second_cossim, queries, p_normed_embs, p_nodes, k), self.pairs)
              # li_results = []
              # partial_func = partial(self._analyse_second_cossim, queries, p_normed_embs, p_nodes, k)
              # for result in tqdm(p.imap(partial_func, self.pairs), total=len(self.pairs)):
              #   li_results.append(result)
        else:
          li_results = []
          for pair in tqdm(self.pairs, desc="Comparing embeddings"):
            li_results.append(self._analyse_second_cossim(queries, p_normed_embs, p_nodes, k, pair))

        results = np.asarray(li_results)

        if save is True:
            np.save(os.path.join(save_path, f"{self.file_prefix}_{k}nn_2nd_order_cossim"), results)

        return nodes, results

    def nearest_neighbors_queries(self, nodes, k, save_path, return_embeddings=False):
        """Uses a ball tree to compute the nearest neighbors in the embedding space"""
        queries = {}
        normed_embs = {}
        # Normalizing the embeddings to be able to use distance as a proxy for cosine similarity
        # BallTree from sklearn is used to compute the neighbors efficiently
        if return_embeddings:
            for emb in tqdm(self.embeddings, desc="Querying nearest neighbors"):
                normed_embs[emb] = normalize(self.read_embedding(emb), norm='l2', copy=False)
                ball_tree = BallTree(normed_embs[emb], leaf_size=40)
                queries[emb] = ball_tree.query(normed_embs[emb][nodes, :], k=k + 1, return_distance=False).tolist()
        else:
            for emb in tqdm(self.embeddings, desc="Querying nearest neighbors"):
                normalized_embedding = normalize(self.read_embedding(emb), norm='l2',
                                                 copy=False)
                ball_tree = BallTree(normalized_embedding, leaf_size=40)
                print("Starting a query")
                # Query the k+1 nearest neighbors, because a node will always be the closest neighbor to itself
                queries[emb] = ball_tree.query(normalized_embedding[nodes, :], k=k + 1, return_distance=False).tolist()
                print("Finish a query")

        # Save the computed neighbors to be able to skip the computation
        self.save_pickle(queries, save_path, self.file_prefix + "_" + str(k) + "nns")

        if return_embeddings:
            return queries, normed_embs
        else:
            return queries

    def sample_nodes(self, k):
        """
        Sample unique nodes of an embedding
        Args:
            k: int, number of nodes to sample
        Returns:
            numpy array of node ids of length k if k is smaller than the number of nodes available.
            Otherwise, all available nodes are returned.
        """
        vertices = np.arange(self.num_vertices)
        np.random.shuffle(vertices)
        return vertices[:min(k, self.num_vertices)]


    def cossim_analysis(self, save_path):
        """ Computes aligned cosine similarity values. Internally performs orthogonal transformation (Procrustes
        problem) between two embeddings and saves transformation matrices as well as vector of resulting errors
        """

        # Set up file naming
        results_suffix = "aligned_cossim"

        # Read the embeddings
        normed_embs = {}
        for emb in tqdm(self.embeddings, desc="Reading embeddings"):
            normed_embs[emb] = normalize(
                    self.read_embedding(emb)[np.arange(self.num_vertices)], norm='l2',
                    copy=False)

        # Do the analysis
        emb_ind = -1
        results = []
        for pair in tqdm(self.pairs, desc="Comparing embeddings"):

            # only update first embedding if it does not change
            if emb_ind != pair[0]:
                emb_ind = pair[0]

            W1 = normed_embs[emb_ind]

            # transform W2 into W1 using procrustes matrix
            Q, _ = orthogonal_procrustes(normed_embs[pair[1]], normed_embs[pair[0]], check_finite=False)
            W2 = normed_embs[pair[1]].dot(Q)

            # Do 1-cosine to get the actual cosine similarity instead of cosine difference
            pair_results = np.array([1-cosine(W1[i], W2[i]) for i in range(self.num_vertices)])
            results.append(pair_results)

        results = np.asarray(results)
        np.save(os.path.join(save_path, f"{self.file_prefix}_{results_suffix}"), results)
        return np.arange(self.num_vertices), results

    def unaligned_cosine_analysis(self, save_path):
        """ Computes regular cosine similarity values, without alignment, between embedding pairs
        """

        # Set up file naming
        results_suffix = "cossim"

        # Read the embeddings
        normed_embs = {}
        for emb in tqdm(self.embeddings, desc="Reading embeddings"):
            normed_embs[emb] = normalize(
                    self.read_embedding(emb)[np.arange(self.num_vertices)], norm='l2',
                    copy=False)

        # Do the analysis
        emb_ind = -1
        results = []
        for pair in tqdm(self.pairs, desc="Comparing embeddings"):

            # only update first embedding if it does not change
            if emb_ind != pair[0]:
                emb_ind = pair[0]

            # Get our two embedding matrics
            W1 = normed_embs[emb_ind]
            W2 = normed_embs[pair[1]]

            # Do 1-cosine to get the actual cosine similarity instead of cosine distance
            pair_results = np.array([1-cosine(W1[i], W2[i]) for i in range(self.num_vertices)])
            results.append(pair_results)

        results = np.asarray(results)
        np.save(os.path.join(save_path, f"{self.file_prefix}_{results_suffix}"), results)
        return np.arange(self.num_vertices), results

    def pairwise_distance(self, nodes, save_path, norm=False):
        """Calculate the pairwise distance between all our embeddings. Save these to the given file path"""

        results_suffix = "euclidean_distance"

        # Read the embeddings
        normed_embs = {}
        for emb in tqdm(self.embeddings, desc="Reading embeddings"):
            # Option for calculating distance between normalized embeddings
            if norm:
              normed_embs[emb] = normalize(
                      self.read_embedding(emb)[np.arange(self.num_vertices)], norm='l2',
                      copy=False)
            else:
              normed_embs[emb] = self.read_embedding(emb)[np.arange(self.num_vertices)]

        # Do the analysis
        emb_ind = -1
        results = []
        for pair in tqdm(self.pairs, desc="Comparing embeddings"):

            # only update first embedding if it does not change
            if emb_ind != pair[0]:
                emb_ind = pair[0]

            # Get the distance between the two matrices of embeddings and add to results array
            results.append(np.linalg.norm(normed_embs[emb_ind] - normed_embs[pair[1]], axis=1))

        results = np.asarray(results)
        np.save(os.path.join(save_path, f"{self.file_prefix}_{results_suffix}"), results)


    def _get_nodes(self, nodes, num_samples, append):
        """
        Handles getting nodes for the experiments.
        Args:
            nodes: list, node ids
            num_samples: int, how many nodes should be sampled
            append: bool, whether to append sampled nodes to specified nodes
        Returns:
            numpy array of (sampled) node ids
        """
        if nodes is None:
            nodes = self.sample_nodes(num_samples)
        elif append is True:
            # allows specified nodes to be taken twice
            nodes.extend(self.sample_nodes(num_samples))
        return np.asarray(nodes)

    def save_pickle(self, obj, save_path, file_name):
        if save_path is None:
            save_path = os.getcwd()
        if file_name is None:
            # generate name of report from embedding input: use name of first embedding file without number information
            file_name = self.file_prefix
        with open(os.path.join(save_path, f"{file_name}.pickle"), "wb") as f:
            pickle.dump(obj, f)

    def get_combinations(self):
        return self.pairs

    def get_vertex_count(self):
        return self.num_vertices

    # The original code reorders the nodes by ID, but our nodes have the same ordering so there is no need to reorder
    def read_embedding(self, path):
      node_embedding = torch.load(path, map_location=torch.device('cpu')).detach().numpy()
      return node_embedding


# Code adapted from https://github.com/SGDE2020/embedding_stability/blob/master/similarity_tests/similarity_tests.py
# Expects the the file names in the embedding directory to be of the form model-name_seed_emb.pt
def run_metric_calculations(embedding_dir, results_dir, tests, num_nodes=-1, knn_size=20, load_knn=False, kload_size=None,
                            num_processes=4, cossim_sec_indices=None):
    """
    Run specified similarity tests.
    Params:
        embedding_dir:
        file_prefix: the prefix for files for saving results
        tests: list of str, tests that will be conducted
        num_nodes: int, number of nodes of a graph that are used in the tests (just the number of nodes in the graph)
        knn_size: int, neighborhood size for knn test
        load_knn: bool, whether a stored knn matrix should be loaded
        nodeinfo_dir = str, path to directory where nodeinfo is stored (tables)
        results_dir = str, path to directory where results will be saved to
    """

    # Create results directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    t = time()
    time_and_memory_log = f'{results_dir}/time_memory_log.txt'
    with open(time_and_memory_log, 'a') as file:
        file.write(f'Logging time and memory usage')
        file.write('\n')
        file.write("Allocation at the start is : torch.cuda.memory_allocated: %fGB" % (
                    torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        file.write('\n')
        file.write(f'RAM memory % usage at the start is: {psutil.virtual_memory()[2]}')
        file.write('\n')
        file.write(f'RAM memory amount usage at the start is: {psutil.virtual_memory()[3]}')
        file.write('\n')

    # Computes a list of all _emb.pt files in all directories.
    # Assumes the directoriesybeing used only contains results from one architecture and data set.
    # fnames = sorted([f for f in os.listdir(embedding_dir) if f.endswith("_emb.pt")])
    fnames = sorted([embedding_dir + f for f in os.listdir(embedding_dir) if f.endswith("_emb.pt")])
    print(fnames)

    if len(fnames) <= 1:
        print(f"Did not find any embeddings for under directory {embedding_dir}")

    else:
        # To be compatible with the original source code, we need a list of nodes indices we will be using in our metric computations
        # We use all the nodes
        nodes = [i for i in range(num_nodes)]

        # Allow us to use specific nodes for the second order cossine similarity as it can be very computational expensive
        cossim_nodes = nodes if cossim_sec_indices is None else cossim_sec_indices

        if load_knn and kload_size is None:
            kload_size = knn_size

        # Start tests
        comp = Comparison(embeddings=fnames, num_nodes=num_nodes, file_prefix="geometric_")
        if "cossim" in tests:
            print("Executing cosine similarity")
            comp.cossim_analysis(save_path=results_dir)
        if "jaccard" in tests:
            print("Executing jaccard score")
            comp.jaccard_similarity(
                nodes=nodes, append=False, k=knn_size,
                load=load_knn, kload_size=kload_size, save=True, save_path=results_dir, num_processes=num_processes
            )
        if "2ndcos" in tests:
            print("Executing second order cosine similarity")
            comp.second_order_cosine_similarity(
                nodes=cossim_nodes, append=False, k=knn_size,
                save=True, save_path=results_dir, num_processes=num_processes, load=load_knn
            )
        if "unalign_cossim" in tests:
            print("Executing cosine similarity")
            comp.unaligned_cosine_analysis(save_path=results_dir)
        if "dist" in tests:
            print("Executing euclidean distance")
            comp.pairwise_distance(nodes=nodes, save_path=results_dir)

    with open(time_and_memory_log, 'a') as file:
        file.write(f'Time taken for run_metric_calculations on {tests} is: {time() - t}')
        file.write('\n')


def read_and_average_metric(file_suffix, class_labels, num_classes):
    results_arr = np.load(file_path+model_directory+"/results/"+model_file_prefix+"_"+file_suffix+".npy", mmap_mode='r')
    results_flat = results_arr.flatten()
    # Compute the 5 number summary for the data across all classes, better than mean and std for potentially skewed distribution
    q1, q2, q3 = np.percentile(results_flat, [25,50,75])
    min, max = results_flat.min(), results_flat.max()
    # Print results
    print("TOTAL STATS")
    print("q1=" + str(q1) + ", q2=" + str(q2) + ", q3=" + str(q3) + ", min=" + str(min) + ", max=" + str(max))

    for i in range(num_classes):
      indices = np.where(class_labels==i)[0]
      # Filter the results by class
      class_vals = results_arr[:, indices]
      # Compute stats for the class
      q1_c, q2_c, q3_c = np.percentile(class_vals, [25,50,75])
      min_c, max_c = class_vals.min(), class_vals.max()
      # Print results
      print("CLASS STATS " + str(i))
      print("q1=" + str(q1_c), ", q2=" + str(q2_c) + ", q3=" + str(q3_c) + ", min=" + str(min_c) + ", max=" + str(max_c))