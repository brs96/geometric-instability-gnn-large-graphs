import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch import optim
from log_embeddings_and_predictions import update_stats, save_training_info, save_final_results, save_predictions
from torcheval.metrics.functional import multiclass_f1_score

def train_gnn(DATASET, X, edge_indices, y, mask, model, optimiser, device):
    model.train()
    # Put data on device
    X = X.to(device)
    edge_indices = edge_indices.to(device)
    y = y.to(device)
    # mask = mask.to(device)
    # Train
    optimiser.zero_grad()
    y_out, _ = model(X, edge_indices)
    y_hat = y_out[mask]
    # print("y_hat: ", y_hat)
    # print("y: ", y.to(torch.float))

    # Proteins
    if DATASET == "ogbn-proteins":
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y.to(torch.float))
    else:
        loss = F.cross_entropy(y_hat, y)

    loss.backward()
    optimiser.step()
    return loss.data


# Training loop using subgraph batching from paper 'Inductive Representation Learning on Large Graphs' https://arxiv.org/pdf/1706.02216.pdf
def train_gnn_subgraph(data_batch, model, optimiser, device):
    total_loss = 0
    for batch in data_batch:
        # Put batch in device
        batch = batch.to(device)
        # Do training loop
        batch_size = batch.batch_size
        optimiser.zero_grad()
        y_out, _ = model(batch.x, batch.edge_index)
        y_out = y_out[:batch_size]
        batch_y = batch.y[:batch_size]
        batch_y = torch.reshape(batch_y, (-1,))
        loss = F.cross_entropy(y_out, batch_y)
        loss.backward()
        optimiser.step()
        # Keep a running total of the loss
        total_loss += float(loss)

    # Get the average loss across all the batches
    loss = total_loss / len(data_batch)
    return loss


def evaluate_gnn(X, edge_indices, y, mask, model, num_classes, device):
    model.eval()
    # Put data on device
    X = X.to(device)
    edge_indices = edge_indices.to(device)
    y = y.to(device)
    mask = mask.to(device)
    # Evaluate
    with torch.no_grad():
        y_out, node_embeddings = model(X, edge_indices)
    y_hat = y_out[mask]
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    num_total = len(y)
    accuracy = 100.0 * (num_correct / num_total)

    # calculate per class accuracy
    values, counts = torch.unique(y_hat[y_hat == y.data], return_counts=True)
    per_class_counts = torch.zeros(num_classes)
    # make sure per_class_counts is on the correct device
    per_class_counts = per_class_counts.to(device)
    # allocate the number of counts per class
    for i, x in enumerate(values):
        per_class_counts[x] = counts[i]
    # find total number of data points per class in the split
    total_per_class = torch.bincount(y.data)
    per_class_accuracy = torch.div(per_class_counts, total_per_class)

    return accuracy, per_class_accuracy, node_embeddings, y_hat

def evaluate_ogbn_proteins(X, edge_indices, y, mask, model):
    y_out, node_embeddings = model(X, edge_indices)
    evaluator = Evaluator(name='ogbn-proteins')

    rocauc = evaluator.eval({
        'y_true': y,
        'y_pred': y_out[mask],
    })['rocauc']

    return node_embeddings, rocauc, y_out

# Training loop
def train_eval_loop_gnn(MODEL, DATASET, model, epochs, edge_indices, train_x, train_y, train_mask, valid_x, valid_y, valid_mask,
                        test_x, test_y, test_mask, num_classes, seed, file_path, device, subgraph_batches=None):
    optimiser = optim.Adam(model.parameters())
    training_stats = None
    # Training loop
    for epoch in range(epochs):
        # If subgraph batching is not provided, use the full graph for training. Otherwise use subgraph batch training regime
        if subgraph_batches is None:
            train_loss = train_gnn(DATASET, train_x, edge_indices, train_y, train_mask, model, optimiser, device)
        else:
            train_loss = train_gnn_subgraph(subgraph_batches, model, optimiser, device)

        if (epoch % 50 == 0 or epoch == (epochs - 1)) and DATASET != 'ogbn-proteins':
            # Calculate accuracy on full graph
            train_acc, train_class_acc, _, _ = evaluate_gnn(train_x, edge_indices, train_y, train_mask, model,
                                                            num_classes,
                                                            device)
            valid_acc, valid_class_acc, _, _ = evaluate_gnn(valid_x, edge_indices, valid_y, valid_mask, model,
                                                            num_classes,
                                                            device)
            print(
                f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
            print("Per class train accuracy: ", train_class_acc)
            print("Per class val accuracy: ", valid_class_acc)
            # store the loss and the accuracy for the final plot
            epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch': epoch}
            training_stats = update_stats(training_stats, epoch_stats)

    # Lets look at our final test performance
    # Only need to get the node embeddings once, take from the training evaluation call
    # Also store all final predictions y_hat on all nodes.
    if DATASET != 'ogbn-proteins':
        test_acc, test_class_acc, test_node_embeddings, test_y_hat = evaluate_gnn(test_x, edge_indices, test_y, test_mask, model,
                                                                 num_classes, device)
        multiclass_f1 = multiclass_f1_score(test_y, test_y_hat, num_classes=num_classes, average=None)
        print(f"Test node embeddings shape is: {test_node_embeddings.shape}")
        print(f"Our final test accuracy for model {MODEL} is: {test_acc:.3f}")
        print("Final per class accuracy on test set: ", test_class_acc)
        print("Final multiclass f1 score per class on test set: ", multiclass_f1)
    else:
        test_node_embeddings, rocauc, test_y_hat = evaluate_ogbn_proteins(test_x, edge_indices, test_y, test_mask, model)

    # Save training stats if on final iteration of the run
    save_training_info(training_stats, test_node_embeddings, file_path, str(seed))
    save_predictions(test_y_hat, file_path, "predictions" + "_" + str(seed))
    # Save final results
    if DATASET != 'ogbn-proteins':
        final_results_list = [seed, test_acc, test_class_acc, train_class_acc, valid_class_acc]
        # final_results_list = [seed, test_acc, test_class_acc, train_class_acc, valid_class_acc, multiclass_f1]
    else:
        final_results_list = [seed, rocauc, 0, 0, 0]

    save_final_results(final_results_list, file_path, seed, "_final_accs")
    # Save final model weights incase we want to do further inference later
    torch.save(model.state_dict(), file_path + MODEL + "_" + str(seed) + "_model.pt")
    print("Finished training, training_stats is: ")
    print(training_stats)
    return training_stats