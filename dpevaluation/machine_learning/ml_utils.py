#!/usr/bin/python3.7

from dpevaluation.utils.log import log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
try:
    import torch
except ImportError:
    pass

# seed for srg
random_state = 1337

def split_and_normalize_data(lib: str, data, benchmark_data, target: str, batch_size):
    # We want deterministic shuffle for synthetic data and benchmark data
    data_size = data.shape[0]
    data = data.sample(data_size, random_state=random_state)
    # Separate the target value/the label (y) from the features (x)
    label = data[target]
    train_data = data.drop([target], axis=1)
    # Divide the data into train and test set
    x_train, x_test, y_train, y_test = train_test_split(
        train_data,
        label,
        test_size=0.2,
        shuffle=False)
    # Synthetic test
    if benchmark_data is not None:
        '''
        Ex:
        Synthetic dataset: train_s=80, test_s=20
        Benchmark dataset: train_b=80 test_b=20
        We want train_s + test_b to evaluate the trained model on the
        benchmark dataset.

        We do this only for synthetic data testing, else: split and 
        normalize the given dataset.
        '''
        benchmark_data = benchmark_data.sample(
            data_size, random_state=random_state)
        benchmark_label = benchmark_data[target]
        benchmark_train_data = benchmark_data.drop([target], axis=1)
        # Divide the benchmark data into train and test set
        _, x_test, _, y_test = train_test_split(
            benchmark_train_data,
            benchmark_label,
            test_size=0.2,
            shuffle=False)
    # Save the input shape for the model
    input_shape = x_train.shape
    # Normalize data
    scaler = StandardScaler()
    norm_x_train = pd.DataFrame(scaler.fit_transform(x_train))
    norm_x_test = pd.DataFrame(scaler.transform(x_test))
    # Build dataloaders for Opacus
    if lib == "opacus":
        train_loader, test_loader = build_dataloaders(
            norm_x_train, y_train, norm_x_test, y_test, batch_size)
        return train_loader, test_loader, input_shape
    else:
        return norm_x_train, y_train, norm_x_test, y_test, input_shape


def build_dataloaders(x_train, y_train, x_test, y_test, _batch_size):
    # Wrap normalized data as PyTorch tensors
    tensor_x_train = \
        torch.tensor(x_train.to_numpy(dtype="float"), dtype=torch.float)
    tensor_x_test = \
        torch.tensor(x_test.to_numpy(dtype="float"), dtype=torch.float)
    # Ensure label has the right shape
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).view(-1, 1)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float).view(-1, 1)
    # Construct a TensorDataset
    train_datasets = torch.utils.data.TensorDataset(tensor_x_train, y_train)
    test_datasets = torch.utils.data.TensorDataset(tensor_x_test, y_test)
    # Generate a DataLoder by using TensorDataset
    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets, batch_size=_batch_size, shuffle=True)
    # Keep labels unshuffled if we want to draw confusion matrix at the end
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets, batch_size=_batch_size, shuffle=False)

    return train_loader, test_loader


def load_datasets(synthetic_test: bool, meta, dataset, benchmark_datasets, trace=False):
    if synthetic_test:
        config = meta.load_synthetic_configuration_with_fallback(dataset)
        dataset = meta.load_synthetic_data(dataset)
        # # Check if there are nan values in synthetic dataset
        # if dataset.isnull().values.any():
        #     log("error", "Synthetic dataset has nan values")
        #     log("debug", "Synthetic dataset config file: %s" % config)
        #     raise Exception("Synthetic dataset has nan values")
        # Load matching benchmark dataset
        benchmark_dataset = benchmark_datasets[dataset.shape[0]]

        if dataset.shape != benchmark_dataset.shape:
            if trace:
                log("warn", "Synthetic dataset shape: %s does not match benchmark dataset shape %s" %
                    (dataset.shape, benchmark_dataset.shape))
                log("debug", "Synthetic dataset config file: %s" % config)
            # Remove any columns from synthetic dataset that are not listed in metadata
            dataset = dataset[dataset.columns.intersection(meta.column_names)]
            # Match benchmark dataset with synthetic dataset
            benchmark_dataset = pd.DataFrame(
                benchmark_dataset, columns=dataset.columns)
    else:
        benchmark_dataset = None

    return dataset, benchmark_dataset, config if synthetic_test else None


def get_datasets(synthetic_test: bool, meta):
    '''
    We use benchmark_datasets for making predictions with model trained
    in synthetic datasets.
    '''
    if synthetic_test:
        benchmark_datasets, _ = meta.split_dataset(return_dict=True)
        datasets = __filter_synthetic_datasets(meta, False)
    else:
        benchmark_datasets = None
        datasets, _ = meta.split_dataset()

    return datasets, benchmark_datasets


def __filter_synthetic_datasets(meta, trace=False):
    '''
    Remove synthetic datasets that do not match size of benchmark datasets
    specified in metadata (dataset_sizes)
    '''
    filtered = []

    w_o = meta.synthetic_datasets_without_results()
    best = meta.best_performing_synthetic_datasets()


    best_performing_without_results = list(set(best) & set(w_o))

    print(best_performing_without_results)

    for ds in best_performing_without_results:
        dataset = meta.load_synthetic_data(ds)
        if dataset.shape[0] in meta.dataset_sizes:
            filtered.append(ds)
        elif trace:
            log("error", "Synthetic dataset size: %s did not match any benchmark dataset sizes: %s" %
                (dataset.shape[0], str(meta.dataset_sizes)[1:-1]))
            config = meta.load_synthetic_configuration_with_fallback(ds)
            log("warn", "Synthetic dataset config file: %s" % config)

    return filtered


def log_info(mode: str, ds, epsilon, noise_multiplier, norm_clip):
    print()
    if mode == "synthetic":
        log("info", "Running %s: size=%d, ε=%.4f" %
            (mode, ds.shape[0], epsilon))
    elif mode == "non_private":
        log("info", "Running %s: size=%d" % (mode, ds.shape[0]))
    else:
        s = "Running %s: size=%d, ε=%.4f"
        if noise_multiplier is not None:
            s += ", nm=%.4f, norm_clip=%.1f" % (noise_multiplier, norm_clip)

        log("info", s % (mode, ds.shape[0], epsilon))


def log_losses(desc: str, losses: dict, s=""):
    # losses = {"mae": [first epoch loss, ..., last epoch loss]}
    for name, loss in losses.items():
        try:
            l = loss[-1]
        except Exception:
            l = loss
        s += name + "=%.4f " % l
    log("debug", "%s: %s" % (desc, s))
