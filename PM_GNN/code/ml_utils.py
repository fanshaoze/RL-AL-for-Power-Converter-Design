import pprint

import torch
import torch.nn.functional as F

import numpy as np
import math

from easydict import EasyDict

from model_different_gnn_encoders import  PT_GNN, Serial_GNN, LOOP_GNN
import copy


def rse(y,yt):

    assert(y.shape==yt.shape)

    if len(y)==0:
        return 0,0
    var=0
    m_yt=yt.mean()
#    print(yt,m_yt)
    for i in range(len(yt)):
        var+=(yt[i]-m_yt)**2
    print("len(y)",len(y))
    var = var/len(y)
    mse=0
    for i in range(len(yt)):
        mse+=(y[i]-yt[i])**2
    mse = mse/len(y)
    # print("var: ", var)
    # print("mse: ",mse)
    rse=mse/(var+0.0000001)

    rmse=math.sqrt(mse/len(yt))

#    print(rmse)

    return rse,mse


def initialize_model(model_index, gnn_nodes, gnn_layers, pred_nodes, nf_size, ef_size, device, output_size=1):
    #初始化模型
    args = EasyDict()
    args.len_hidden = gnn_nodes
    args.len_hidden_predictor = pred_nodes
    args.len_node_attr = nf_size
    args.len_edge_attr = ef_size
    args.gnn_layers = gnn_layers
    args.use_gpu = False
    args.dropout = 0
    args.output_size = output_size

    if model_index == 1:
        model = PT_GNN(args).to(device)
        return model
    elif model_index == 2:
        model = Serial_GNN(args).to(device)
        return model
    elif model_index == 3:
        model = LOOP_GNN(args).to(device)
        return model
    else:
        assert ("Invalid model")

    #选择model

def train(train_loader, val_loader, model, n_epoch, batch_size, num_node, device, model_index, optimizer,gnn_layers):
    train_perform=[]

    min_val_loss=100

    for epoch in range(n_epoch):
    
    ########### Training #################
        
        train_loss=0
        n_batch_train=0
    
        model.train()

        for i, data in enumerate(train_loader):
                 data.to(device)
                 L=data.node_attr.shape[0]
                 B=int(L/num_node)
#                 print(L,B,data.node_attr)
                 node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                 if model_index == 0:
                     edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                 else:
                     edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                     edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])
 
                 adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                 y=data.label
                 n_batch_train=n_batch_train+1
                 optimizer.zero_grad()
                 if model_index == 0:
                      out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                 else:
                      out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device), gnn_layers))
 
                 out=out.reshape(y.shape)
                 assert(out.shape == y.shape)
                 loss=F.mse_loss(out, y.float())

#                 loss=F.binary_cross_entropy(out, y.float())
                 loss.backward()
                 optimizer.step()
        
                 train_loss += out.shape[0] * loss.item()
        
        if epoch % 1 == 0:
                 print('%d epoch training loss: %.3f' % (epoch, train_loss/n_batch_train/batch_size))

                 n_batch_val=0
                 val_loss=0

#                epoch_min=0
                 model.eval()

                 for data in val_loader:

                     n_batch_val+=1

                     data.to(device)
                     L=data.node_attr.shape[0]
                     B=int(L/num_node)
                     node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                     if model_index == 0:
                         edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                     else:
                         edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                         edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])

                     adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                     y=data.label

                     n_batch_train=n_batch_train+1
                     optimizer.zero_grad()
                     if model_index == 0:
                          out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                     else:
                          out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device),gnn_layers))

                     out=out.reshape(y.shape)
                     assert(out.shape == y.shape)
#                     loss=F.binary_cross_entropy(out, y.float())
                     loss=F.mse_loss(out,y.float())
                     val_loss += out.shape[0] * loss.item()
                 val_loss_ave=val_loss/n_batch_val/batch_size
                 

                 if val_loss_ave<min_val_loss:
                    model_copy=copy.deepcopy(model)
                    print('lowest val loss', val_loss_ave)
                    epoch_min=epoch
                    min_val_loss=val_loss_ave

                 if epoch-epoch_min>5:
                    #print("training loss:",train_perform)
                    print("training loss minimum value:", min(train_perform))
                    print("training loss average value:", np.mean(train_perform))

                    return model_copy, min(train_perform), np.mean(train_perform)


        train_perform.append(train_loss/n_batch_train/batch_size)

    return model, min(train_perform), np.mean(train_perform)


def test(test_loader, model, num_node, model_index, device,gnn_layers):
        model.eval()
        accuracy=0
        n_batch_test=0
        gold_list=[]
        out_list=[]
        analytic_list = []


        TPR=0
        FPR=0
        count=0
        for data in test_loader:
             data.to(device)
             L=data.node_attr.shape[0]
             B=int(L/num_node)
             node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
             if model_index == 0:
                 edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
             else:
                 edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                 edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])

             adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
             y=data.label.cpu().detach().numpy()

             n_batch_test=n_batch_test+1
             if model_index==0:
                  out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
             else:
                  out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device),gnn_layers)).cpu().detach().numpy()
             out=out.reshape(y.shape)
             assert(out.shape == y.shape)
             out=np.array([x for x in out])
             # Shun: the following needs to be disabled for reg_both
             # It shouldn't affect reg_eff, reg_vout, etc.
             #gold=np.array(y.reshape(-1))
             gold=np.array([x for x in y])

             gold_list.extend(gold)
             out_list.extend(out)

             L=len(gold)
             np.set_printoptions(precision=2,suppress=True)
#
             """
             out_round=[int(i>0.8) for i in out]
             TP=0
             FN=0
             FP=0
             TN=0
             for i in range(len(gold)):
                 if gold[i]==1 and out_round[i]==1:
                     TP+=1
                 elif gold[i]==1 and out_round[i]==0:
                     FN+=1
                 elif gold[i]==0 and out_round[i]==1:
                     FP+=1
                 else:
                     TN+=1
              """
#             print(TP/(TP+FN),FP/(TN+FP))
#             TPR+=TP/(TP+FN)
#             FPR+=FP/(TN+FP)
#             count+=1
#        print("Average erro:",TPR/count,FPR/count)
#        print("Predicted:",out[0:128])
#        print("True     :",gold[0:128])
#        print("Error    :",len([i for i in abs(out-gold) if i > 0.5])/len(out))
#        print(gold_list)
        result_bins = compute_errors_by_bins(np.reshape(out_list,-1),np.reshape(gold_list,-1),[(-0.1,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.1)])

        final_rse, final_mse = rse(np.reshape(out_list,-1),np.reshape(gold_list,-1))
        print("Final RSE:", final_rse)
        return final_rse,result_bins


def compute_errors_by_bins(pred_y:np.array, true_y:np.array, bins):
    """
    Divide data by true_y into bins, report their rse separately
    :param pred_y: model predictions (of the test data)
    :param true: true labels (of the test data)
    :param bins: a list of ranges where errors in these ranges are computed separately
                 e.g. [(0, 0.33), (0.33, 0.66), (0.66, 1)]
    :return: a list of rses by bins
    """
    results = []

    for range_from, range_to in bins:
        # get indices of data in this range
        indices = np.nonzero(np.logical_and(range_from <= true_y, true_y < range_to))

        if len(indices) > 0:
            temp_rse, temp_mse = rse(pred_y[indices], true_y[indices])
            results.append(math.sqrt(temp_mse))
            # print('data between ' + str(range_from) + ' ' + str(range_to))
            # pprint.pprint(list(zip(pred_y[indices], true_y[indices])))
        else:
            print('empty bin in the range of ' + str(range_from) + ' ' + str(range_to))

    return results
