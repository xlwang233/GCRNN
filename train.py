import argparse
import collections
import torch

import lib.metrics as module_metric
import model.dcrnn_model as module_arch
from parse_config import ConfigParser
from trainer.dcrnn_trainer import DCRNNTrainer
from lib import utils


def main(config):
    logger = config.get_logger('train')

    graph_pkl_filename = 'data/sensor_graph/top10_graph.pkl'
    # _, _, adj_mat = utils.load_graph_data(graph_pkl_filename)
    graphs = utils.load_dynamic_graph(graph_pkl_filename)  # (952, 207, 207)
    data = utils.load_dataset(dataset_dir='data/METR-LA',
                              graphs=graphs,
                              batch_size=config["arch"]["args"]["batch_size"],
                              test_batch_size=config["arch"]["args"]["batch_size"])
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print((k, v.shape))

    train_data_loader = data['train_loader']
    val_data_loader = data['val_loader']

    # get number of iterations per epoch for progress bar
    num_train_iteration_per_epoch = data['train_loader'].num_batch
    num_val_iteration_per_epoch = data['val_loader'].num_batch

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize('loss', module_metric, **{"scaler": data['scaler']})
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = DCRNNTrainer(model, loss, metrics, optimizer,
                           config=config,
                           data_loader=train_data_loader,
                           valid_data_loader=val_data_loader,
                           lr_scheduler=lr_scheduler,
                           len_epoch=num_train_iteration_per_epoch,
                           val_len_epoch=num_val_iteration_per_epoch)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch DCRNN')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)


