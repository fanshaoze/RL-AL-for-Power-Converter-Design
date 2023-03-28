import os

import numpy as np
import torch
from torch.nn import Linear, MSELoss

from topo_data import Autopo, split_balance_data

from ml_utils import train, test, rse, initialize_model
import argparse


if __name__ == '__main__':

# ======================== Arguments ==========================#
    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-y_select', type=str, default='reg_vout', help='define target label')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=100, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=100, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=100, help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=5, help='number of layer')
    parser.add_argument('-model_index', type=int, default=1, help='model index')
    parser.add_argument('-threshold', type=float, default=0, help='classification threshold')
    parser.add_argument('-ncomp', type=int, default=5, help='# components')
    parser.add_argument('-train_rate', type=float, default=0.2, help='# components')
 
    parser.add_argument('-retrain', action='store_true', default=True, help='force retrain model')
    parser.add_argument('-seed', type=int, default=0, help='random seed')

    args = parser.parse_args()


    ncomp=args.ncomp
    train_rate=args.train_rate
    path=args.path
    y_select=args.y_select
    data_folder_3='../2_dataset/'+y_select+'_3'
    data_folder_5='../2_dataset/'+y_select+'_5'    
    batch_size=args.batch_size
    n_epoch=args.n_epoch
    th=args.threshold
 
# ======================== Data & Model ==========================#

    dataset = Autopo(data_folder_3,path,y_select,3)
    train_loader, val_loader, test_loader = split_balance_data(dataset, batch_size,0.7,0.15,0.15)

    # set random seed for training
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    nf_size=4
    ef_size=3
    nnode=7
    if args.model_index==0:
        ef_size=6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    model = initialize_model(model_index=args.model_index,
                             gnn_nodes=args.gnn_nodes,
                             gnn_layers=args.gnn_layers,
                             pred_nodes=args.predictor_nodes,
                             nf_size=nf_size,
                             ef_size=ef_size,
                             device=device)

    postfix = '' if device.type == 'cuda' else '_cpu'
    pt_filename = y_select + postfix + '.pt'

    if os.path.exists(pt_filename) and not args.retrain:
        print('loading model from pt file')

        model_state_dict, _ = torch.load(pt_filename)
        model.load_state_dict(model_state_dict)
    else:
        print('training')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = MSELoss(reduction='mean').to(device)
        model = train(train_loader=train_loader,
                      val_loader=val_loader,
                      model=model,
                      n_epoch=n_epoch,
                      batch_size=batch_size,
                      num_node=nnode,
                      device=device,
                      model_index=args.model_index,
                      optimizer=optimizer)

        # save model and test data
        torch.save((model.state_dict(), test_loader), y_select + '.pt')
    print("3 comp model on 3 comp data:")
 
    test(test_loader=test_loader, model=model, num_node=nnode, model_index=args.model_index, device=device)

    dataset5 = Autopo(data_folder_5,path,y_select,5)

    train_loader_5, val_loader_5, test_loader_5 = split_balance_data(dataset5, batch_size,train_rate,0.1,0.1)
    print("3 comp model on 5 comp data:")
 
    test(test_loader=test_loader_5, model=model, num_node=nnode, model_index=args.model_index, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = MSELoss(reduction='mean').to(device)

    model_5 = model

#    model_5 = initialize_model(model_index=args.model_index,
#                             gnn_nodes=args.gnn_nodes,
#                             gnn_layers=args.gnn_layers,
#                             pred_nodes=args.predictor_nodes,
#                             nf_size=nf_size,
#                             ef_size=ef_size,
#                             device=device)
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#    criterion = MSELoss(reduction='mean').to(device)
 

    model_5 = train(train_loader=train_loader_5,
                  val_loader=val_loader_5,
                  model=model_5,
                  n_epoch=n_epoch,
                  batch_size=batch_size,
                  num_node=nnode,
                  device=device,
                  model_index=args.model_index,
                  optimizer=optimizer)


    print("5 comp model on 5 comp data:")
    test(test_loader=test_loader_5, model=model_5, num_node=nnode, model_index=args.model_index, device=device)
    print("5 comp model on 3 comp data:") 
    test(test_loader=test_loader, model=model_5, num_node=nnode, model_index=args.model_index, device=device)


            
    
#    np.random.seed(args.seed)
#    torch.manual_seed(args.seed)
#    torch.cuda.manual_seed(args.seed)

#    nf_size=4
#    ef_size=3
#    nnode=6
#    if args.model_index==0:
#        ef_size=6
#
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    data = dataset5[0].to(device)

#    model = initialize_model(model_index=args.model_index,
#                             gnn_nodes=args.gnn_nodes,
#                             gnn_layers=args.gnn_layers,
#                             pred_nodes=args.predictor_nodes,
#                             nf_size=nf_size,
#                             ef_size=ef_size,
#                             device=device)

#    postfix = '' if device.type == 'cuda' else '_cpu'
#    pt_filename = y_select + postfix + '.pt'



