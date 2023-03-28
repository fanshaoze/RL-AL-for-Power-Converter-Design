import pprint
import time
import os

import wandb
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn

import numpy as np

from dataset import get_loader

from build_vocab import Vocabulary
from Optim import NoamOpt, get_std_opt, get_huggingface_opt
from Models import get_model, create_masks, GPModel
from pytorchtools import EarlyStopping
import gpytorch
from test_model import test

from process_data import get_pred_and_y
from transformer_args import get_transformer_args
from util import feed_random_seeds, compute_rse, compute_classification_error


def train_epoch(model, training_data, target, optimizer, lr_schedular, device, criterion, likelihood=None,
                gp=None, use_gp=True):
    ''' Epoch operation in training phase'''

    gp.train()
    model.train()
    likelihood.train()

    all_pred = []
    all_y = []

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        path, eff, vout, duty, valid, padding_mask = map(lambda x: x.to(device), batch)

        optimizer.zero_grad()

        pred, final = model(path, duty, padding_mask)
        if use_gp:
            pred = gp(pred)
        else:
            pred = final.squeeze()

        pred, y = get_pred_and_y(eff, vout, valid, pred, target, device)

        # all_pred.append(pred)
        # all_y.append(y)

        # backward
        if use_gp:
            loss = -criterion(pred, y)  # criterion is a likelihood
            pred = pred.mean  # only need mean to compute errors
        else:
            loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        all_pred.append(pred)
        all_y.append(y)

    lr_schedular.step()

    all_pred, all_y = map(lambda x: np.array(torch.cat(x).detach().cpu()), [all_pred, all_y])
    if target == 'valid':
        error, metrics = compute_classification_error(pred=all_pred, y=all_y)
    else:
        error = compute_rse(pred=all_pred, y=all_y)

    return error


def eval_epoch(model, validation_data, target, device, vocab, criterion, likelihood, gp, use_gp=True):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    likelihood.eval()
    gp.eval()

    all_pred = []
    all_y = []

    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            path, eff, vout, duty, valid, padding_mask = map(lambda x: x.to(device), batch)

            pred, final = model(path, duty, padding_mask)
            if use_gp:
                # pred = gp(pred)
                pred = gp(pred).mean
            else:
                pred = final.squeeze()

            pred, y = get_pred_and_y(eff=eff, vout=vout, valid=valid, pred=pred, target=target, device=device)

            # note keeping
            # if use_gp:
            #     # only need mean for computing mse
            #     pred = pred.mean

            all_pred.append(pred)
            all_y.append(y)

    all_pred, all_y = map(lambda x: np.array(torch.cat(x).cpu()), [all_pred, all_y])
    if target == 'valid':
        error, metrics = compute_classification_error(pred=all_pred, y=all_y)
    else:
        error = compute_rse(pred=all_pred, y=all_y)
    return error


def train(model, training_data, validation_data, optimizer, lr_schedular, args, vocab, criterion, likelihood, gp,
          testing_data=None):
    """
    :return: rse on training set and test set
    """
    early_stopping_with_saving = EarlyStopping(patience=args.patience, verbose=True, args=args)

    for epoch_i in range(args.epoch):
        if early_stopping_with_saving.early_stop:
            break

        lr = optimizer.param_groups[0]['lr']
        print('Epoch {}, lr {}'.format(epoch_i, lr))

        start = time.time()

        train_loss = train_epoch(
            model, training_data, args.target, optimizer, lr_schedular, args.device, criterion=criterion,
            likelihood=likelihood,
            gp=gp, use_gp=args.use_gp)

        print('  - (Training)   loss: {loss: 8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss,
            elapse=(time.time() - start) / 60))

        start = time.time()

        valid_loss = eval_epoch(model, validation_data, args.target, args.device, vocab, criterion=criterion,
                                likelihood=likelihood, gp=gp, use_gp=args.use_gp)

        print('  - (Validation) loss: {loss: 8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=valid_loss,
            elapse=(time.time() - start) / 60))

        if epoch_i != 0:
            early_stopping_with_saving(valid_loss, model, epoch_i, gp, likelihood, training_data.dataset)

        logs = {'train_rse': train_loss,
                'valid_rse': valid_loss,
                'lr': lr,
                'patience': early_stopping_with_saving.counter}

        if epoch_i % 10 == 0 and epoch_i > 0:
            # evaluate on test sets every 5 steps. no need to evaluate on test sets in every step
            _, test_rse = test(args, model=model, gp=gp, final_call=False, testing_data=testing_data)
            logs.update({'test_rse': test_rse})

        if args.use_log:
            wandb.log(logs)

        if args.save_model:
            checkpoint = {
                'model': model.state_dict(),
                'gp_model': gp.state_dict(),
                'likelihood': likelihood.state_dict(),
                'settings': args,
                'epoch': epoch_i}
            torch.save(checkpoint, args.save_model + '.latest.chkpt')

    return train_loss, valid_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_trainable_parameter_norm(model):
    with torch.no_grad():
        norm = sum(
            [p.norm(1).item()
             for p in model.parameters() if p.requires_grad])
    return norm


def main(args=None, training_data=None, validation_data=None, testing_data=None, transformer=None, gp=None,
         likelihood=None, epoch=None, patience=None, retrain=False,
         ):
    """
    Main function, if transformer, gp and likelihood are provided, initialize using them,
    otherwise start training from scratch.

    :param patience:
    :param epoch:
    :param likelihood:
    :param gp:
    :param transformer:
    :param training_data: training data
    :param testing_data: testing data
    :param validation_data: validation data
    :param args: use args here if not None, otherwise use command line
    @param retrain:
    """
    if args is None:
        args = get_transformer_args()
    elif isinstance(args, list):
        # if some arguments and values are provided in a list, initialize using these values
        # used by re-training the transformer
        args = get_transformer_args(args)
    feed_random_seeds(args.seed)

    if args.save_model:
        model_path = args.save_model.split("/")
        model_path = "/".join(model_path[:-1])
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    pprint.pprint(vars(args))

    if epoch:
        args.epoch = epoch
    if patience:
        args.patience = patience

    if args.data_train or training_data:
        print("======================================start training======================================")
        vocab = Vocabulary()

        vocab.load(args.vocab)

        args.vocab_size = len(vocab)

        # Build data loader
        data_loader_training = get_loader(data=training_data or args.data_train,
                                          vocab=vocab,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          ground_truth=args.ground_truth,
                                          num_workers=args.num_workers,
                                          max_seq_len=args.max_seq_len,
                                          attribute_len=args.attribute_len
                                          )

        data_loader_dev = get_loader(data=validation_data or args.data_dev,
                                     vocab=vocab,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     ground_truth=args.ground_truth,
                                     max_seq_len=args.max_seq_len,
                                     attribute_len=args.attribute_len
                                     )

        # gp_initial_data = getRawData(args.data_train, vocab, args.max_seq_len)

        batch = next(iter(data_loader_training))
        path, eff, vout, duties, valids, padding_mask = map(lambda x: x.to(args.device), batch)

        transformer = transformer or get_model(args, load_weights=args.load_weights)
        transformer = transformer.to(args.device)

        # total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = 0
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
                total_params += param.data.numel()
        print(f"The number of parameters is {total_params}")

        with torch.no_grad():
            gp_initial_data, final = transformer(path, duties, padding_mask)
        gp_initial_data = gp_initial_data.detach()
        gp = gp or GPModel(gp_initial_data)
        likelihood = likelihood or gpytorch.likelihoods.GaussianLikelihood()
        if args.cuda:
            gp = gp.cuda()
            likelihood = likelihood.cuda()

        # optimizer = get_std_opt(gp, args, likelihood, transformer)
        optimizer, lr_schedular = get_huggingface_opt(gp, args, likelihood, transformer)

        if args.use_gp:
            criterion = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=len(data_loader_training.dataset))
        elif args.target == 'valid':
            # it's a classification task
            criterion = nn.BCELoss(reduction="mean")
        else:
            # it's a regression task
            criterion = nn.MSELoss(reduction="mean")

        train_rse, valid_rse = train(transformer, data_loader_training, data_loader_dev, optimizer, lr_schedular, args,
                                     vocab, criterion, likelihood, gp, testing_data=testing_data)

    if args.data_test:
        print("======================================start testing==============================")
        # args.pretrained_model = args.save_model
        _, test_rse = test(args, model=transformer, gp=gp, final_call=None, testing_data=testing_data)
    else:
        test_rse = None

    # for transformer, return these for possible further training
    return transformer, gp, likelihood, data_loader_training.dataset, train_rse, valid_rse, test_rse


if __name__ == '__main__':
    main()
