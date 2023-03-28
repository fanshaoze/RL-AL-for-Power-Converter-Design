''' Translate input text with trained model. '''
import os

import torch
import torch.utils.data

import wandb
from tqdm import tqdm

from dataset import get_loader
from build_vocab import Vocabulary
from Models import get_model, GPModel
import numpy as np
import time
import gpytorch

from plot_distribution import plot_distribution
from process_data import get_pred_and_y
from transformer_args import get_transformer_args
from util import compute_rse, visualize, feed_random_seeds, compute_classification_error


def test(opt, model=None, gp=None, final_call=True, testing_data=None):
    """
    Evaluate model and gp if set, otherwise load them from opt.pretrained_model
    If opt.validity_model is not None, first check if topo is valid, then make prediction

    :param plot_file: plot ground truth vs prediction in this file
    :param final_call: if true, save final performances to log and do plotting.
    Otherwise, it's a functional call during the training process
    """
    if isinstance(opt.vocab, str):
        vocab = Vocabulary()
        vocab.load(opt.vocab)
    else:
        vocab = opt.vocab

    data_loader = get_loader(data=testing_data or opt.data_test,
                             vocab=vocab,
                             batch_size=opt.batch_size,
                             shuffle=False,  # don't shuffle, so we know which topologies are which
                             ground_truth=opt.test_ground_truth,
                             max_seq_len=opt.max_seq_len,
                             attribute_len=opt.attribute_len
                             )

    if model is None:
        # if model or gp is not set, read from file named pretrained_model
        checkpoint = torch.load(opt.pretrained_model + '.chkpt')

        model = get_model(opt, load_weights=True)
        model = model.to(opt.device)

        if opt.use_gp:
            gp_para = checkpoint["gp_model"]

            gp = GPModel(gp_para["variational_strategy.inducing_points"])
            gp.load_state_dict(gp_para)
            gp = gp.to(opt.device)

    if opt.pretrained_validity_model is not None:
        validity_model = get_model(opt, pretrained_model=opt.pretrained_validity_model, load_weights=True)
        validity_model = validity_model.to(opt.device)
    else:
        validity_model = None

    model.eval()
    if opt.use_gp:
        gp.eval()

    all_pred = []
    all_y = []

    inf_times = []
    with torch.no_grad(): # gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        for batch in tqdm(data_loader, mininterval=2, desc='  - (Test)', leave=False):
            path, eff, vout, duty, valid, padding_mask = map(lambda x: x.to(opt.device), batch)


            start = time.time()  # start timing here
            # print('b-path:', path)
            # print('b-pad:', padding_mask)

            last_layer, final = model(path, duty, padding_mask)
            if opt.use_gp:
                # pred = gp(last_layer)
                pred = gp(last_layer).mean
            else:
                pred = final.squeeze(dim=1)

            # print('b pred after model', pred)


            if validity_model:
                last_layer, final = validity_model(path, duty, padding_mask)
                valid_mask = final.squeeze()
            else:
                valid_mask = torch.ones_like(pred)  # if no validity mask, assume they are all valids

            # if validity classifier thinks it's invalid, set to 0
            pred = torch.where((valid_mask > 0.5), pred, torch.zeros_like(pred))

            pred, y = get_pred_and_y(eff=eff, vout=vout, valid=valid, pred=pred, target=opt.target, device=opt.device)

            # if opt.use_gp:
            #     # only need mean for computing mse
            #     pred = pred.mean

            # if the output range should be in 0 to 1:
            # unnecessary since sigmoid is added
            pred = torch.clamp(pred, 0., 1.)
            # print('b pred after clamp', pred)

            end = time.time()  # timing ends here


            all_pred.append(pred)
            all_y.append(y)

            inf_times.append(end - start)
            # if final_call and wandb.run is not None:
            #     wandb.log({'inference_time': end - start})

    # convert to numpy arrays
    all_pred, all_y = map(lambda x: np.array(torch.cat(x).cpu()), [all_pred, all_y])

    if final_call and opt.plot_outliers:
        outlier_indices = np.where(np.abs(all_pred - all_y) > 0.5)[0]
        outlier_folder = f"{opt.data_test}_outliers"

        print("processing outlier data")
        if len(outlier_indices) > 200:
            print(f"too many outliers (total {len(outlier_indices)}). only showing the first 200 outliers")
            outlier_indices = outlier_indices[:200]

        if not os.path.exists(outlier_folder):
            os.makedirs(outlier_folder)

        for outlier_index in tqdm(outlier_indices):
            datum = data_loader.dataset.data[outlier_index]
            visualize(list_of_node=datum['list_of_nodes'],
                      list_of_edge=datum['list_of_edges'],
                      info=f"duty = {datum['duty']}, ground truth eff = {datum['eff']:.4f}, predicted eff = {all_pred[outlier_index]:.4f}",
                      filename=f"{outlier_folder}/{outlier_index}")

    if opt.target == 'valid':
        # it's a classification task
        error, metrics = compute_classification_error(pred=all_pred, y=all_y)

        if opt.use_log and final_call:
            for metric_name, metric_value in metrics.items():
                wandb.run.summary[metric_name] = metric_value
    else:
        error = compute_rse(pred=all_pred, y=all_y)

        if opt.use_log and final_call:
            # save all y and predictions
            plot_distribution(all_y=all_y.tolist(), all_pred=all_pred.tolist(), target=opt.target, use_wandb=True)

    if final_call:
        print(f"error is: {error}")

        mean_inf_time = np.mean(inf_times)
        print("avg inference time is: ", mean_inf_time)
        if wandb.run is not None:
            wandb.run.summary["avg_inference_time"] = mean_inf_time
            wandb.run.summary["final_test_rse"] = error

    return all_pred, error


if __name__ == "__main__":
    args = get_transformer_args()
    feed_random_seeds(args.seed)

    if args.use_log:
        wandb.init(project="surrogate_model",
                   name=args.data_test,
                   config=vars(args))

    test(args)
