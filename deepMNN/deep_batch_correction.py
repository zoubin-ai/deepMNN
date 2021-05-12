import torch
import operator
import numpy as np
import scanpy as sc
from .model import Net
from fbpca import pca
from annoy import AnnoyIndex
from intervaltree import IntervalTree
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, vstack


# Default parameters.
ALPHA = 0.10
APPROX = True
BATCH_SIZE = 1024
BETA1 = 0
BETA2 = 0.001
DECAY_STEP_SIZE = 20
EPOCHS_WO_IM = 10
EPS = 1e-2
KNN = 20
LR_DECAY_FACTOR = 0.8
LR = 1e-1
MAX_EPOCHS = 200
MIN_LR = 1e-6
N_BLOCKS = 2
PLOT_LOSS = False
VERBOSE = 2
WEIGHT_DECAY = 1e-4


# Exact nearest neighbors search.
def nn(ds1, ds2, knn=KNN, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, knn=KNN, metric='manhattan', n_trees=10):
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Populate a table (in place) that stores mutual nearest neighbors
# between datasets.
def fill_table(table, i, curr_ds, datasets, base_ds=0, knn=KNN, approx=APPROX):
    curr_ref = np.concatenate(datasets)
    if approx:
        match = nn_approx(curr_ds, curr_ref, knn=knn)
    else:
        match = nn(curr_ds, curr_ref, knn=knn, metric_p=1)

    # Build interval tree.
    itree_ds_idx = IntervalTree()
    itree_pos_base = IntervalTree()
    pos = 0
    for j in range(len(datasets)):
        n_cells = datasets[j].shape[0]
        itree_ds_idx[pos:(pos + n_cells)] = base_ds + j
        itree_pos_base[pos:(pos + n_cells)] = pos
        pos += n_cells

    # Store all mutual nearest neighbors between datasets.
    for d, r in match:
        interval = itree_ds_idx[r]
        assert(len(interval) == 1)
        j = interval.pop().data
        interval = itree_pos_base[r]
        assert(len(interval) == 1)
        base = interval.pop().data
        if not (i, j) in table:
            table[(i, j)] = set()
        table[(i, j)].add((d, r - base))
        assert(r - base >= 0)
        
# Fill table of alignment scores.
def find_alignments_table(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE, prenormalized=False):
    if not prenormalized:
        datasets = [ normalize(ds, axis=1) for ds in datasets ]

    table = {}
    for i in range(len(datasets)):
        if len(datasets[:i]) > 0:
            fill_table(table, i, datasets[i], datasets[:i], knn=knn,
                       approx=approx)
        if len(datasets[i+1:]) > 0:
            fill_table(table, i, datasets[i], datasets[i+1:],
                       knn=knn, base_ds=i+1, approx=approx)
    # Count all mutual nearest neighbors between datasets.
    matches = {}
    table1 = {}
    if verbose > 1:
        table_print = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i >= j:
                continue
            if not (i, j) in table or not (j, i) in table:
                continue
            match_ij = table[(i, j)]
            match_ji = set([ (b, a) for a, b in table[(j, i)] ])
            matches[(i, j)] = match_ij & match_ji

            table1[(i, j)] = (max(
                float(len(set([ idx for idx, _ in matches[(i, j)] ]))) /
                datasets[i].shape[0],
                float(len(set([ idx for _, idx in matches[(i, j)] ]))) /
                datasets[j].shape[0]
            ))
            if verbose > 1:
                table_print[i, j] += table1[(i, j)]

    if verbose > 1:
        print(table_print)
        return table1, table_print, matches
    else:
        return table1, None, matches

# Find the matching pairs of cells between datasets.
def find_alignments(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE, alpha=ALPHA, prenormalized=False):
    table1, _, matches = find_alignments_table(
        datasets, knn=knn, approx=approx, verbose=verbose,
        prenormalized=prenormalized,
    )

    alignments = [ (i, j) for (i, j), val in reversed(
        sorted(table1.items(), key=operator.itemgetter(1))
    ) if val > alpha ]

    return alignments, matches

def update_lr(optim, epoch, init_lr, min_lr=MIN_LR, decay_step_size=DECAY_STEP_SIZE, lr_decay_factor=LR_DECAY_FACTOR):
    """ stepwise learning rate calculator """
    exponent = int(np.floor((epoch + 1) / decay_step_size))
    lr = init_lr * np.power(lr_decay_factor, exponent)
    if lr < min_lr:
        optim.param_groups[0]['lr'] = min_lr
    else:
        optim.param_groups[0]['lr'] = lr
    print('Learning rate = %.7f' % optim.param_groups[0]['lr'])

def training_step(net, optim, batch, PCs, beta1=BETA1, beta2=BETA2):
    size = int(batch.shape[0]/2)
    corrected_batch = net(batch)
    corrected_batch_pca = torch.matmul(corrected_batch, PCs)
    batch1 = batch[:size, :]
    batch2 = batch[size:, :]
    corrected_batch1 = corrected_batch[:size, :]
    corrected_batch2 = corrected_batch[size:, :]
    corrected_batch1_pca = corrected_batch_pca[:size, :]
    corrected_batch2_pca = corrected_batch_pca[size:, :]
    loss1 = torch.sum(torch.mean(torch.abs(corrected_batch2_pca - corrected_batch1_pca), axis=1))
    loss2 = torch.sum(torch.mean(torch.abs(corrected_batch1 - corrected_batch2), axis=1))
    loss3 = torch.sum(torch.mean(torch.abs(corrected_batch1 - batch1), axis=1)) + torch.sum(torch.mean(torch.abs(corrected_batch2 - batch2), axis=1))
    loss = loss1 + beta1 * loss2 + beta2 * loss3

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def correct_scanpy(adatas,
                   n_blocks=N_BLOCKS,
                   lr=LR,
                   weight_decay=WEIGHT_DECAY,
                   batch_size=BATCH_SIZE,
                   eps=EPS,
                   epochs_wo_im=EPOCHS_WO_IM,
                   max_epochs=MAX_EPOCHS,
                   plot_loss=PLOT_LOSS,
                   decay_step_size=DECAY_STEP_SIZE
):
    if len(adatas) == 1:
        return adatas

    best_loss = 10e10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_pca = [adata.copy().obsm['X_pca'] for adata in adatas]
    datasets = [adata.X.toarray() for adata in adatas]

    PCs = adatas[0].varm['PCs']
    PCs_tensor = torch.Tensor(PCs.copy()).to(device=device)

    # find MNN
    alignments, matches = find_alignments(datasets_pca, knn=KNN, approx=APPROX, verbose=VERBOSE, alpha=ALPHA)
    match = []
    for i, j in alignments:
        base_i, base_j = 0, 0
        for k in range(i):
            base_i += adatas[k].shape[0]
        for k in range(j):
            base_j += adatas[k].shape[0]
        match.extend([ (a + base_i, b + base_j) for a, b in matches[(i, j)] ])
    match = np.array([[a,b] for a, b in match])

    input_dim = datasets[0].shape[1]
    net = Net(input_dim, n_blocks, device)
    # from torchsummary import summary
    # summary(net, (50,))

    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    datasets = np.concatenate(datasets)

    data_loader = torch.utils.data.DataLoader(list(range(len(match))),
                                              batch_size=batch_size,
                                              shuffle=True)

    losses = []
    net.train(True)
    for epoch in range(max_epochs):
        batch_losses = []
        for _, batch in enumerate(data_loader):
            batch1 = datasets[match[batch.numpy(),0],:]
            batch2 = datasets[match[batch.numpy(),1],:]
            batch_data = np.concatenate((batch1, batch2))
            batch_data = torch.Tensor(batch_data).to(device=device)
            batch_loss = training_step(net, optim, batch_data, PCs_tensor)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)
        if epoch_loss < best_loss - eps:
            best_loss = epoch_loss
            epoch_counter = 0
        else:
            epoch_counter += 1

        print('Epoch {}, loss: {:.3f}, counter: {}'.format(epoch, epoch_loss, epoch_counter))

        update_lr(optim, epoch, init_lr=lr, decay_step_size=decay_step_size)

        if epoch_counter == epochs_wo_im:
            break

    print('Finished training')

    if plot_loss:
        plt.plot(losses[10:])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    net.train(False)
    
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(datasets)),
                                              batch_size=batch_size,
                                              shuffle=False)
    corrected = []
    for _, batch in enumerate(data_loader):
        batch = batch[0].to(device=device)
        corrected += [net(batch).detach().cpu().numpy()]
    corrected = np.concatenate(corrected)

    from anndata import AnnData

    new_adatas = []
    base = 0
    for i in range(len((adatas))):
        adata = AnnData(corrected[base:base+adatas[i].shape[0], :])
        adata.var_names = adatas[i].var_names
        adata.obs = adatas[i].obs
        new_adatas.append(adata)
        base += adatas[i].shape[0]
        
    return new_adatas