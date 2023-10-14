import argparse
import os
import re
import pickle
import io

import numpy as np
import seaborn as sns
import pandas as pd
import torch
from matplotlib import pyplot as plt


def create_plot_sns(data, title, filename, x_val, hue_val, distance=False, jaccard=False):

    color_dict = {"GATv1": "#FF9933", "GraphSAGE": "#FFFF00", "GIN": "#00FFFF",
                  "GCN": "#3333FF", "DirGATv1": "#FFD39B", "DirGraphSAGE": "#CAFF70", "DirGCN": "#808080"}

    light_gray = ".8"
    dark_gray = ".15"
    sns.set(context="notebook", style="whitegrid", font_scale=1,
            rc={"axes.edgecolor": light_gray, "xtick.color": dark_gray,
                "ytick.color": dark_gray, "xtick.bottom": True,
                "font.size": 8, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 15, "legend.fontsize": 4,
                "ytick.labelsize": 15, "axes.linewidth": 1,
                "xtick.minor.width": 0.5, "xtick.major.width": 0.5,
                "ytick.minor.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.7,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "grid.linewidth": 0.5
                })

    g = sns.catplot(data=data, x=x_val, y="Similarity", kind="box", hue=hue_val, legend=False,
                    palette=color_dict, aspect=2, whis=1000000)
    # height=width/2,
    g.set_ylabels(title, fontsize='17')
    g.set_xlabels("")
    g.axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.45, -0.05), fancybox=False, shadow=False, ncol=5,
                        fontsize='17')
    if distance:
        g.axes[0, 0].set_ylim(0.0, 180)
    elif jaccard:
        g.axes[0, 0].set_ylim(0.0, 1.0)
    else:
        g.axes[0, 0].set_ylim(0.0, 1.0)

    g.set_titles("")
    g.savefig("plots/" + filename + ".pdf", bbox_inches="tight")



def create_plot_sns_with_acc(gram_data, test_acc_data, title, filename, x_val, hue_val, distance=False, jaccard=False):

    color_dict = {"GATv1": "#FF9933", "GraphSAGE": "#FFFF00", "GIN": "#00FFFF",
                  "GCN": "#3333FF", "DirGATv1": "#FFD39B", "DirGraphSAGE": "#CAFF70", "DirGCN": "#808080"}

    # light_gray = ".8"
    # dark_gray = ".15"
    # sns.set(context="notebook", style="whitegrid", font_scale=1,
    #         rc={"axes.edgecolor": light_gray, "xtick.color": dark_gray,
    #             "ytick.color": dark_gray, "xtick.bottom": True,
    #             "font.size": 8, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 15, "legend.fontsize": 4,
    #             "ytick.labelsize": 15, "axes.linewidth": 1,
    #             "xtick.minor.width": 0.5, "xtick.major.width": 0.5,
    #             "ytick.minor.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.7,
    #             "xtick.major.size": 3,
    #             "ytick.major.size": 3,
    #             "xtick.minor.size": 2,
    #             "ytick.minor.size": 2,
    #             "grid.linewidth": 0.5
    #             })

    g = sns.catplot(data=gram_data, x=x_val, y="Similarity", kind="box", legend=False,
                    palette=color_dict, aspect=2, whis=1000000)
    # height=width/2,
    g.set_ylabels(title, fontsize='17')
    g.set_xlabels("")

    if distance:
        g.axes[0, 0].set_ylim(0.0, 180)
    elif jaccard:
        g.axes[0, 0].set_ylim(0.0, 1.0)
    else:
        g.axes[0, 0].set_ylim(-0.2, 0.8)


    g2 = plt.twinx()
    p2 = sns.scatterplot(data=test_acc_data, x=x_val, y="Test Acc", hue=hue_val, palette=color_dict, legend=False)
    p2.set_ylabel("Test Accuracy", fontsize='17')
    g2.set_ylim(0, 100)

    plt.title("")
    plt.savefig("plots/" + filename + ".pdf", bbox_inches="tight")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    dataset = args.dataset

    pattern = f'_cora/analysis/geometric__20nn_jaccard.npy'

    file_path = f'plots/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if args.metric == "jaccard":

        all_data = []

        for filename in os.listdir('model_results'):
            # Check if the filename matches the pattern
            model_type = filename.split('_')[0]
            if re.search(f'_{args.dataset}', filename):
                file_path = os.path.join('model_results', filename, 'analysis', 'geometric__20nn_jaccard.npy')

                results_arr = np.load(file_path, mmap_mode='r')
                results_flat = results_arr.flatten()
                q1, q2, q3 = np.percentile(results_flat, [25, 50, 75])
                min, max = results_flat.min(), results_flat.max()
                data = pd.DataFrame([[min, dataset, model_type],
                                     [q1, dataset, model_type],
                                     [q2, dataset, model_type],
                                     [q3, dataset, model_type],
                                     [max, dataset, model_type]], columns=["Similarity", "Dataset", "Model"])

                all_data.append(data)

        create_plot_sns(pd.concat(all_data), "20-NN-Jaccard", filename='jaccard', x_val='Dataset', hue_val='Model',
                        distance=False, jaccard=True)

    if args.metric == "2ndcos":

        all_data = []

        for filename in os.listdir('model_results'):
            model_type = filename.split('_')[0]
            if re.search(f'_{args.dataset}', filename):
                file_path = os.path.join('model_results', filename, 'analysis', 'geometric__20nn_2nd_order_cossim.npy')
                results_arr = np.load(file_path, mmap_mode='r')
                results_flat = results_arr.flatten()
                q1, q2, q3 = np.percentile(results_flat, [25, 50, 75])
                min, max = results_flat.min(), results_flat.max()
                data = pd.DataFrame([[min, dataset, model_type],
                                     [q1, dataset, model_type],
                                     [q2, dataset, model_type],
                                     [q3, dataset, model_type],
                                     [max, dataset, model_type]], columns=["Similarity", "Dataset", "Model"])

                all_data.append(data)

        create_plot_sns(pd.concat(all_data), "Second-order cossim", filename='2ndcos', x_val='Dataset', hue_val='Model',
                        distance=False, jaccard=True)

    if args.metric == "gram":

        gram_data = []

        for filename in os.listdir('model_results'):
            model_type = filename.split('_')[0]
            if re.search(f'_{args.dataset}', filename):
                file_path = os.path.join('model_results', filename, 'analysis', 'normalized_centered_ggi.csv')
                print(f'File path is: {file_path}')

                max_num_columns = 0
                last_row_fields = []
                with open(file_path, 'r') as file:
                    for line in file:
                        fields = line.strip().split(',')
                        num_columns = len(fields)

                        if num_columns > max_num_columns:
                            max_num_columns = num_columns
                        last_row_fields = fields

                # Convert the last row fields to a numpy array
                results_arr = np.array(last_row_fields)
                data = pd.DataFrame([[float(results_arr[0]), dataset, model_type],
                                     [float(results_arr[1]), dataset, model_type],
                                     [float(results_arr[2]), dataset, model_type],
                                     [float(results_arr[3]), dataset, model_type],
                                     [float(results_arr[4]), dataset, model_type]], columns=["Similarity", "Dataset", "Model"])

                gram_data.append(data)

        test_acc_data = []

        for filename in os.listdir('model_results'):
            model_type = filename.split('_')[0]
            if re.search(f'_{args.dataset}', filename):
                folder_path = os.path.join('model_results', filename)
                print(folder_path)
                print(model_type)
                single_model_type_results = np.array([], dtype=float)
                for pkl in os.listdir(folder_path):
                    if re.search(f'_final_accs', pkl):
                        pkl_path = os.path.join(folder_path, pkl)
                        print(pkl_path)
                        with open(pkl_path, 'rb') as f:
                            #final_accs = np.load(f, allow_pickle=True)
                            final_accs = CPU_Unpickler(f).load()
                        print('final test accs:  ' + str(final_accs[1].item()))
                        single_model_type_results = np.append(single_model_type_results, final_accs[1].item())
                        print('Current single_model_type_results contains:  ')
                        print(single_model_type_results)

                # q1, q2, q3 = np.percentile(single_model_type_results, [25, 50, 75])
                # min, max = single_model_type_results.min(), single_model_type_results.max()
                df_data = {'Test Acc': single_model_type_results, 'Dataset': dataset, 'Model': model_type}
                data = pd.DataFrame(df_data)
                test_acc_data.append(data)
                print('Current all_data contains:  ')
                print(test_acc_data)

        custom_order = ['GATv1', 'GraphSAGE', 'GCN', 'GIN', 'DirGATv1', 'DirGraphSAGE', 'DirGCN']
        gram_data = sorted(gram_data, key=lambda df: [custom_order.index(val) for val in df['Model']])
        test_acc_data = sorted(test_acc_data, key=lambda df: [custom_order.index(val) for val in df['Model']])

        create_plot_sns_with_acc(pd.concat(gram_data), pd.concat(test_acc_data), "Graph Gram Index", filename=f'{dataset}_gram', x_val='Model', hue_val='Model',
                        distance=False, jaccard=False)
