import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime

from torchmeta.utils.data import BatchMetaDataLoader
from maml.utils import load_dataset, load_model, update_parameters, get_accuracy
import plotly.graph_objects as go
from torch.autograd import Variable



class KLLoss(nn.Module):
    def __init__(self, tempval , alpha , beta):
        super(KLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temp = tempval
    def forward(self, pred, label , target):
        
        one_target = torch.nn.functional.one_hot(target)
        none_target = 1 - one_target
        batch_size , class_size = pred.shape[0] , pred.shape[1]

        T = self.temp
        p_stu = F.softmax(pred / T, dim=1)
        p_tea = F.softmax(label / T, dim=1)
         
        pt_tea , pnt_tea = p_tea[one_target.bool()] , (p_tea * none_target).sum(1)
        pt_stu , pnt_stu = p_stu[one_target.bool()] , (p_stu * none_target).sum(1)

        pnct_stu = F.softmax(pred[none_target.bool()].reshape(batch_size , class_size-1 ) / T , dim =1)
        pnct_tea = F.softmax(label[none_target.bool()].reshape(batch_size , class_size-1 ) / T , dim =1)



        #target_data = target_data + 10 ** (-7)
        #target_data = F.softmax(label / T, dim=1)
        # pt_tea_ng = Variable(pt_tea.data, requires_grad=False)
        # pnt_tea_ng = Variable(pnt_tea.data, requires_grad=False)
        # pnct_tea_ng = Variable(pnct_tea.data, requires_grad=False)

        # print("pt_tea" , pt_tea.shape)
        # print("pnt_tea" , pnt_tea.shape)
        # print("pnct_tea" , pnct_tea.shape)
        # print()
        
        tckd = kl_div(pt_tea , pt_stu , T) + kl_div(pnt_tea , pnt_stu , T)
        nckd = kl_div(pnct_tea , pnct_stu , T)

        loss = (self.alpha * tckd) + (self.beta * nckd) * T**2
        return loss

def kl_div(target, predict , T):
    if len(target) == 2: #2 dimension
        return ((target * (target.log() - predict.log())).sum(1).sum() / target.size()[0])
    else:
        return ((target * (target.log() - predict.log())).sum() / target.size()[0])

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def main_val(args, mode, iteration=None):
    dataset = load_dataset(args, mode)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model.to(device=args.device)
    model.train()
    criterion_kl = KLLoss(args.temp , args.Alpha , args.Beta)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # To control outer update parameter
    # If you want to control inner update parameter, please see update_parameters function in ./maml/utils.py
    freeze_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    learnable_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
    if args.outer_fix:
        meta_optimizer = torch.optim.Adam([{'params': freeze_params, 'lr': 0},
                                           {'params': learnable_params, 'lr': args.meta_lr}])
    else:
        meta_optimizer = torch.optim.Adam([{'params': freeze_params, 'lr': args.meta_lr},
                                           {'params': learnable_params, 'lr': args.meta_lr}])

    if args.meta_train:
        total = args.train_batches
    elif args.meta_val:
        total = args.valid_batches
    elif args.meta_test:
        total = args.test_batches

    loss_logs_sub1, accuracy_logs_sub1, loss_logs_sub2, accuracy_logs_sub2, loss_logs_ensem, accuracy_logs_ensem = [], [], [], [],[], []
    step_size = OrderedDict()
    for name, _ in model.named_parameters():
        if 'classifier' in name:
            step_size[name] = args.classifier_step_size
        else:
            step_size[name] = args.extractor_step_size


    mean_cossim=0
    mean_head=0
    mean_conv4=0
    mean_conv3=0
    mean_conv2=0
    mean_conv1=0



    # Training loop
    with tqdm(dataloader, total=total, leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if args.centering:
                fc_weight_mean = torch.mean(model.classifier.weight.data, dim=0)
                model.classifier.weight.data -= fc_weight_mean

            model.zero_grad()

            support_inputs, support_targets = batch['train']
            support_inputs = support_inputs.to(device=args.device)
            support_targets = support_targets.to(device=args.device)

            query_inputs, query_targets = batch['test']
            query_inputs = query_inputs.to(device=args.device)
            query_targets = query_targets.to(device=args.device)


            # for subnet 1
            outer_loss_sub1 = torch.tensor(0., device=args.device)
            accuracy_sub1 = torch.tensor(0., device=args.device)

            # for subnet 2
            outer_loss_sub2 = torch.tensor(0., device=args.device)
            accuracy_sub2 = torch.tensor(0., device=args.device)

            # for ensemble
            outer_loss_ensem = torch.tensor(0., device=args.device)
            accuracy_ensem = torch.tensor(0., device=args.device)





            classifier_grad_sim = []
            similarlity4  = []
            similarlity3  = []
            similarlity2  = []
            similarlity1  = []

            for task_idx, (support_input, support_target, query_input, query_target) in enumerate(
                    zip(support_inputs, support_targets, query_inputs, query_targets)):
                params = None
                for _ in range(args.inner_update_num):
                    support_features, support_logit, fixed_support_logit = model(support_input, params=params)
                    inner_loss = F.cross_entropy(support_logit, support_target)

                    model.zero_grad()
                    params, head_grad, list_grad4, list_grad3, list_grad2, list_grad1 = update_parameters(model,
                                                                                                          inner_loss,
                                                                                                          extractor_step_size=args.extractor_step_size,
                                                                                                          classifier_step_size=args.classifier_step_size,
                                                                                                          fixed_classifier_step_size=args.fixed_classifier_step_size,fixed_last_step_size=args.fixed_last_step_size,
                                                                                                          first_order=args.first_order,
                                                                                                          GRIL=args.GRIL,
                                                                                                          norm=args.norm,
                                                                                                          scale=args.scale)


                query_features, query_logit, fixed_query_logit = model(query_input, params=params)
                outer_loss_sub1 = outer_loss_sub1+  F.cross_entropy(query_logit, query_target) 
                outer_loss_sub2 = outer_loss_sub2+  F.cross_entropy(fixed_query_logit, query_target)
                outer_loss_ensem = outer_loss_ensem+F.nll_loss((F.softmax(fixed_query_logit, 1)+F.softmax(query_logit,1)).log(), query_target,weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean')


                with torch.no_grad():
                    accuracy_sub1 += get_accuracy(query_logit, query_target)
                    accuracy_sub2 += get_accuracy(fixed_query_logit, query_target)
                    accuracy_ensem += get_accuracy((F.softmax(fixed_query_logit ,-1)+F.softmax(query_logit ,-1))/2, query_target)





            # sub_network1
            outer_loss_sub1.div_(args.batch_size)
            accuracy_sub1.div_(args.batch_size)
            loss_logs_sub1.append(outer_loss_sub1.item())
            accuracy_logs_sub1.append(accuracy_sub1.item())
            #

            # sub_network2
            outer_loss_sub2.div_(args.batch_size)
            accuracy_sub2.div_(args.batch_size)
            loss_logs_sub2.append(outer_loss_sub2.item())
            accuracy_logs_sub2.append(accuracy_sub2.item())
            #

            # ensemble
            outer_loss_ensem.div_(args.batch_size)
            accuracy_ensem.div_(args.batch_size)
            loss_logs_ensem.append(outer_loss_ensem.item())
            accuracy_logs_ensem.append(accuracy_ensem.item())
            #

            total_loss = outer_loss_sub1 +outer_loss_sub2



            if args.meta_train:
                meta_optimizer.zero_grad()
                total_loss.backward()
                meta_optimizer.step()

            postfix = {'mode': mode, 'iter': iteration, 'acc': round(accuracy_sub1.item(), 5)}
            pbar.set_postfix(postfix)
            if batch_idx + 1 == total:
                break



    # Save best model
    if args.meta_val:
        filename = os.path.join(args.output_folder, args.dataset + '_' + args.save_name, 'logs', 'logs.csv')
        valid_logs = list(pd.read_csv(filename)['valid_accuracy'])

        max_acc = max(valid_logs)
        curr_acc = np.mean(accuracy_logs_sub1)

        if max_acc < curr_acc:
            filename = os.path.join(args.output_folder, args.dataset + '_' + args.save_name, 'models', 'best_val_acc_model.pt')
            with open(filename, 'wb') as f:
                state_dict = model.state_dict()
                torch.save(state_dict, f)

    return loss_logs_sub1, accuracy_logs_sub1, loss_logs_sub2, accuracy_logs_sub2, loss_logs_ensem, accuracy_logs_ensem


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset: miniimagenet, tieredimagenet, cub, cars, cifar_fs, fc100, aircraft, vgg_flower')
    parser.add_argument('--model', type=str, help='Model: 4conv, resnet')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')

    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--inner-update-num', type=int, default=1, help='The number of inner updates (default: 1).')
    parser.add_argument('--extractor-step-size', type=float, default=0.5,
                        help='Extractor step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--classifier-step-size', type=float, default=0.5,
                        help='Classifier step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--fixed-last-step-size', type=float, default=0.5,
                        help='Classifier step-size for the gradient step for adaptation (default: 0.5).')

    parser.add_argument('--fixed-classifier-step-size', type=float, default=0.5,
                        help='Classifier step-size for the gradient step for adaptation (default: 0.5).')

    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--blocks-type', type=str, default=None, help='Resnet block type (optional).')

    parser.add_argument('--output-folder', type=str, default='./output/',
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save-name', type=str, default=None, help='Name of model (optional).')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--batch-iter', type=int, default=300,
                        help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 300).')
    parser.add_argument('--train-batches', type=int, default=100,
                        help='Number of batches the model is trained over (i.e., validation save steps) (default: 100).')
    parser.add_argument('--valid-batches', type=int, default=25,
                        help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test-batches', type=int, default=2500,
                        help='Number of batches the model is tested over (default: 2500).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')

    parser.add_argument('--centering', action='store_true', help='Parallel shift operation in the head.')
    parser.add_argument('--ortho-init', action='store_true', help='Use the head from the orthononal model.')
    parser.add_argument('--outer-fix', action='store_true', help='Fix the head during outer updates.')

    parser.add_argument('--GRIL', action='store_true', help='GRIL regularization.')
    parser.add_argument('--norm', type=str, default=None, help='GRIL grad norm option: l2 or max')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='multiplier')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='multiplier')
    parser.add_argument('--mag', type=float, default=0.5,
                        help='multiplier')

    parser.add_argument('--Alpha', type=float, default=0.5,
                        help='tckd')
    parser.add_argument('--Beta', type=float, default=0.5,
                        help='nckd')

    args = parser.parse_args()
    args.save_dir = '{}_{}shot_{}_{}'.format(args.dataset,
                                             args.num_shots,
                                             args.model,
                                             args.save_name)


    args.device = torch.device(args.device)
    model = load_model(args)

    epoch_list = [100, 200, 300]

    s1_error = []
    s1_acc = []
    s2_error = []
    s2_acc = []
    en_error = []
    en_acc = []




    result=pd.DataFrame()
    for idx, e in enumerate(epoch_list):
        best_val_so_far = os.path.join(args.output_folder, args.dataset + '_' + args.save_name, 'models',
                                       f'best_until_{str(e)}.pt')
        model.load_state_dict(
            torch.load(best_val_so_far))
        model.eval()
        if args.ortho_init:
            X = np.random.randn(5, 1600)
            Q = gs(X)

            model.classifier.weight.data = nn.Parameter(Q)

        val_loss_logs_sub1, val_accuracy_logs_sub1, val_loss_logs_sub2, val_accuracy_logs_sub2, val_loss_logs_ensem, val_accuracy_logs_ensem = main_val(args=args, mode='meta_test')
        best_test_error = np.mean(val_loss_logs_sub1)
        best_tess_acc = np.mean(val_accuracy_logs_sub1)


        sub2_loss = np.mean(val_loss_logs_sub2)
        sub2_acc = np.mean(val_accuracy_logs_sub2)
        ensem_loss = np.mean(val_loss_logs_ensem)
        ensem_acc = np.mean(val_accuracy_logs_ensem)


        s1_error.append(best_test_error)
        s1_acc.append(best_tess_acc)
        s2_error.append(sub2_loss)
        s2_acc.append(sub2_acc)
        en_error.append(ensem_loss)
        en_acc.append(ensem_acc)
    nums = [100, 200, 300]

    result["epoch"]=nums
    result["s1_error"]=s1_error
    result["s1_acc"]=s1_acc
    result["s2_error"] = s2_error
    result["s2_acc"] = s2_acc
    result["ensem_error"] = en_error
    result["ensem_acc"] = en_acc

    OutputFoler = os.path.join('./output_DKD', args.dataset + '_' + args.save_name)
    createDirectory(OutputFoler)
    result.to_csv(OutputFoler + "/" +"acc.csv" , index = None)


