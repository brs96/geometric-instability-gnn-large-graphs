import csv
import re
from time import time

import torch
import numpy as np
import os
import psutil
from sklearn.preprocessing import normalize


class GGI:

    def __init__(self, dataset, model, device, NUM_NODES):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.NUM_NODES = NUM_NODES
        if (NUM_NODES == 132534): # proteins
            self.chunk_node_count = 3582  # 37 chunks
        elif (NUM_NODES == 169343): # arxiv, prime number
            self.chunk_node_count = 3500  # 48+1 chunk
        elif (NUM_NODES == 2708): # cora
            self.chunk_node_count = 2708
        elif (NUM_NODES == 2927963): # ogbl-citation2
            self.chunk_node_count = 5000  # 585+1 chunks

    def compute_ggi(self, adj, embeddings, device):
        ggi_list = []

        for emb in embeddings:
            ggi_for_emb = self.compute_embedding_ggi(adj, emb, device)
            ggi_list.append(ggi_for_emb)

        return ggi_list

    def compute_embedding_ggi(self, adj, embedding, device, time_and_memory_log):

        embedding = embedding.to(torch.float32)
        emb_tranpose = embedding.t()

        chunk_count = emb_tranpose.shape[1] // self.chunk_node_count
        remainder = emb_tranpose.shape[1] % self.chunk_node_count
        if (remainder != 0):
            chunk_count += 1

        emb_t_split = torch.split(emb_tranpose, self.chunk_node_count, dim=1)

        del emb_tranpose

        ggi_sum_list = []
        col_position = 0
        for i in range(chunk_count):
            torch.cuda.empty_cache()
            gram_chunk = embedding @ emb_t_split[i]
            ggi_sum_chunk, new_col_pos = self.sparse_dense_mul(adj, gram_chunk, self.chunk_node_count, i, col_position, chunk_count, remainder, device)
            col_position = new_col_pos

            ggi_sum_list.append(ggi_sum_chunk)

        return sum(ggi_sum_list)/adj.nnz()


    def sparse_dense_mul(self, s, d, gap, cursor, col_position, chunk_count, remainder, device):
        rows = s.storage.row()
        cols = s.storage.col()
        row_segment = rows[(cursor * gap <= rows) & (rows < (cursor + 1) * gap)]

        segment_length = row_segment.shape[0]
        new_col_position = col_position + segment_length

        col_segment = cols[col_position: new_col_position]

        segment = torch.stack((torch.sub(row_segment, cursor * gap), col_segment))

        if (cursor != chunk_count-1) | (chunk_count == 1):
            sparse_segment = torch.sparse_coo_tensor(segment, torch.ones(segment.shape[1]), (gap, self.NUM_NODES))
        else:
            # last chunk
            sparse_segment = torch.sparse_coo_tensor(segment, torch.ones(segment.shape[1]), (remainder, self.NUM_NODES))

        dense_segment = sparse_segment.to_dense().t()

        res = torch.mul(dense_segment, d)
        return res.sum(), new_col_position

    def compute_empirical_var(self, ggi_list):
        return torch.std(torch.stack(ggi_list))

    def compute_box_plot(self, ggi_list):
        ggi_np = np.array(ggi_list)
        q1, q2, q3 = np.percentile(ggi_np, [25, 50, 75])
        min_ggi, max_ggi = ggi_np.min(), ggi_np.max()
        return min_ggi, q1, q2, q3, max_ggi


    def center_embedding(self, embedding):
        # Center embeddings columnwise
        emb_centered_by_dim = torch.mean(embedding, dim=0)
        centered_emb = embedding - emb_centered_by_dim
        return centered_emb


    def ggi_over_dataset(self, adj, device, directory, output_path):
        print(f'Using device: {device}')
        pattern = r"\d+\_emb.pt"

        normalized_ggi_list = []

        time_and_memory_log = f'{output_path}/time_memory_log.txt'
        with open(time_and_memory_log, 'a') as file:
            file.write("Allocation at the start is : torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
            file.write('\n')
            file.write(f'RAM memory % usage at the start is: {psutil.virtual_memory()[2]}')
            file.write('\n')

        t = time()
        for filename in os.listdir(directory):
            if re.match(pattern, filename):
                file_path = os.path.join(directory, filename)
                embedding = torch.load(file_path, map_location=torch.device('cpu'))

                embedding = embedding.detach().numpy()

                adjacency = adj.to(torch.float32)
                adjacency.to(device)

                torch_normalized_emb = torch.from_numpy(normalize(embedding, norm='l2', axis=1))
                normalized_centered_ggi = self.compute_embedding_ggi(adjacency, self.center_embedding(torch_normalized_emb), device, time_and_memory_log)

                with open(time_and_memory_log, 'a') as file:
                    file.write(
                        "torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
                    file.write('\n')
                    file.write(f'RAM memory % used: {psutil.virtual_memory()[2]}')
                    file.write('\n')

                normalized_ggi_list.append(normalized_centered_ggi.cpu())

        with open(time_and_memory_log, 'a') as file:
            file.write(f"Time taken to compute ggi and centered_ggi for all embeddings is: {time() - t}")
            file.write('\n')

        normalized_centered_ggi_var = self.compute_empirical_var(normalized_ggi_list)
        normalized_centered_ggi_box_plot = self.compute_box_plot(normalized_ggi_list)
        normalized_centered_ggi_file_path = f'{output_path}/normalized_centered_ggi.csv'

        with open(normalized_centered_ggi_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(normalized_ggi_list)
            writer.writerow([normalized_centered_ggi_var])
            writer.writerow(normalized_centered_ggi_box_plot)
