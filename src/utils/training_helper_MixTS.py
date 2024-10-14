import os
import random
import shutil
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tsaug
import time
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax

from src.models.MultiTaskClassification import NonLinClassifier, MetaModel_AE
from src.models.model import CNNAE
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, remove_empty_dirs,AUMCalculator, consistency_loss
import torch.nn.functional as F

######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns

def replace_threshold_examples(noisy_idx, aum_calculator,args):
  num_threshold_examples = min(int(len(noisy_idx) *args.sample_rate), len(noisy_idx) // args.nbins)
  threshold_data_ids = random.sample(noisy_idx, num_threshold_examples)
  aum_calculator.switch_threshold_examples(threshold_data_ids)
  return threshold_data_ids

######################################################################################################
class clean_noisy_data():
    def __init__(self, args, dataset, clean_index, batch_mu, transform_fn=None):
        self.args = args
        # select clean samples use clean_index
        self.clean_data = (dataset.tensors[0][clean_index], dataset.tensors[1][clean_index], dataset.tensors[2][clean_index], dataset.tensors[3][clean_index])
        noisy_idx = np.setdiff1d(np.arange(args.num_training_samples), clean_index.astype(int))
        self.noisy_data = (dataset.tensors[0][noisy_idx], dataset.tensors[1][noisy_idx], dataset.tensors[2][noisy_idx], dataset.tensors[3][noisy_idx])
        self.batch_mu = batch_mu
        self.transform_fn = transform_fn
        self.clean_size=len(clean_index)
        self.noisy_size=len(noisy_idx)

        self.u_indices = list(range(self.noisy_size))
        # Shuffle the order
        self.u_indices = random.shuffle(self.u_indices)
        self.threshold_examples_ids = set()

    def noisy_len(self):
        return self.noisy_size

    def __len__(self):
        return self.clean_size

    def __getitem__(self, idx):
        assert self.transform_fn is not None
        x_clean=self.clean_data[0][idx]
        y_clean=self.clean_data[1][idx]
        idx_clean=self.clean_data[2][idx]

        noisy_idx=list(range(idx*self.batch_mu,idx*self.batch_mu+self.batch_mu))
        noisy_idx=[i%self.noisy_size for i in noisy_idx]

        noisy_id=noisy_idx

        clean_week, clean_strong = self.transform_fn(x_clean.unsqueeze(0))

        noisy_x=self.noisy_data[0][noisy_id]
        noisy_y=self.noisy_data[1][noisy_id]
        idx_noisy=self.noisy_data[2][noisy_id]
        noisy_week, noisy_strong = self.transform_fn(noisy_x)

        return (x_clean,clean_week, clean_strong , y_clean,idx_clean), (noisy_x,noisy_week, noisy_strong,noisy_y,idx_noisy)


class Transform_week_strong(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, x):
        sample = x.cpu().numpy()
        weak = sample

        if self.args.mixts_aug_type == 'TimeWarp':
            strong =  tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(sample)
        elif self.args.mixts_aug_type == 'GNoise':
            strong = tsaug.AddNoise(scale=0.015).augment(sample)
        elif self.args.mixts_aug_type == 'Convolve':
            strong = tsaug.Convolve(window='flattop',size=10).augment(sample)
        elif self.args.mixts_aug_type == 'Crop':
            strong = tsaug.Crop(size=int(self.args.sample_len*(2/3)),resize=int(self.args.sample_len)).augment(sample)
        elif self.args.mixts_aug_type == 'Drift':
            strong = tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(sample)
        else:
            raise ValueError('augmentation type not found')
        return torch.from_numpy(weak), torch.from_numpy(strong)
def save_model_and_sel_dict(model,args,sel_dict=None):
    model_state_dict = model.state_dict()
    datestr = time.strftime(('%Y%m%d'))
    model_to_save_dir = os.path.join(args.basicpath, 'src', 'model_save', args.dataset)
    if not os.path.exists(model_to_save_dir):
        os.makedirs(model_to_save_dir, exist_ok=True)

    if args.label_noise == -1:
        label_noise = 'inst{}'.format(int(args.ni * 100))
    elif args.label_noise == 0:
        label_noise = 'sym{}'.format(int(args.ni * 100))
    else:
        label_noise = 'asym{}'.format(int(args.ni * 100))
    filename = os.path.join(model_to_save_dir, args.model)
    if sel_dict is not None:
        filename_sel_dict = '{}{}_{}_{}_sel_dict.npy'.format(filename, args.aug, label_noise, datestr)
        np.save(filename_sel_dict, sel_dict)  # save sel_ind
    filename = '{}{}_{}_{}.pt'.format(filename, args.aug, label_noise, datestr)
    torch.save(model_state_dict, filename)  # save model

def test_step(data_loader, model,model2=None):
    model = model.eval()
    if model2 is not None:
        model2 = model2.eval()

    yhat = []
    ytrue = []

    for x, y in data_loader:
        x = x.to(device)

        if model2 is not None:
            logits1 = model(x)
            logits2 = model2(x)
            logits = (logits1 + logits2) / 2
        else:
            logits = model(x)

        yhat.append(logits.detach().cpu().numpy())
        try:
            y = y.cpu().numpy()
        except:
            y = y.numpy()
        ytrue.append(y)

    yhat = np.concatenate(yhat,axis=0)
    ytrue = np.concatenate(ytrue,axis=0)
    y_hat_proba = softmax(yhat[:,:-1], axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)
    accuracy = accuracy_score(ytrue, y_hat_labels)
    f1_weighted = f1_score(ytrue, y_hat_labels, average='weighted')

    return accuracy, f1_weighted




def train_model(model, train_loader, test_loader, args, train_dataset=None, saver=None):
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)

    # Initialize learning history lists
    train_acc_list = []
    train_acc_list_aug = []
    train_avg_loss_list = []
    test_acc_list = []
    test_f1s = []

    # Extract training labels
    y_train = train_dataset.tensors[1]
    y_train_clean = train_dataset.tensors[3]

    # Initialize variables
    features = []
    confident_set_id = np.array([])
    aum_calculator = AUMCalculator(args.delta, int(args.nbins), args.num_training_samples, args.percentile)
    transform_fn = Transform_week_strong(args)
    threshold_data_ids = np.array([])
    classwise_acc_confidence = torch.zeros((args.nbins,)).to(device)

    try:
        loss_all = np.zeros((args.num_training_samples, args.epochs))
        precise_all = np.zeros((args.epochs,2))
        selected_label_confidence = torch.ones(args.num_training_samples, dtype=torch.long) * -1
        selected_label_confidence = selected_label_confidence.to(device)
        conf_num = []

        for e in range(args.epochs):
            if e > args.warmup:
                aum_threshold = aum_calculator.retrieve_threshold()

            # Training step
            if e <= args.warmup:
                train_accuracy, avg_loss, model_new, feature = warmup_CTW(
                    data_loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    transform_fn=transform_fn,
                    loss_all=loss_all,
                    epoch=e,
                    args=args
                )
            else:
                # Create a clean_noisy_data
                clean_noisy_data_set = clean_noisy_data(args, train_dataset, confident_set_id, args.batch_mu, transform_fn=transform_fn)
                clean_noisy_data_loader = DataLoader(clean_noisy_data_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
                train_accuracy, avg_loss, model_new, feature = train_step_CTW(
                    data_loader=clean_noisy_data_loader,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    args=args,
                    aum_calculator=aum_calculator,
                    threshold_data_ids=threshold_data_ids,
                    selected_label_confidence=selected_label_confidence,
                    classwise_acc_confidence=classwise_acc_confidence,
                    loss_all=loss_all,
                    epoch=e
                )

            if e >= args.warmup:
                if e == args.warmup or e % args.switch_sample == 0:
                    threshold_data_ids = replace_threshold_examples(range(args.num_training_samples), aum_calculator, args)
                    aum_threshold = aum_calculator.retrieve_threshold()
                select_samples, _ = clean_selection(args, device, train_loader, feature, aum_calculator, aum_threshold,threshold_data_ids)

                # Calculate the accuracy of clean samples
                clean_acc = accuracy_score(y_train_clean[torch.where(select_samples == 1)[0]], y_train[torch.where(select_samples == 1)[0]])

                noisy_num = len(select_samples) - torch.sum(select_samples)
                clean_num = torch.sum(select_samples)
                batch_mu = math.ceil(noisy_num / clean_num)

                args.batch_mu = batch_mu
                print(f'select num: {clean_num}, clean_acc: {clean_acc}, noisy num: {noisy_num}, batch_mu: {batch_mu}, threshold_num: {len(threshold_data_ids)}')
                confident_set_id = torch.where(select_samples == 1)[0].cpu().numpy()

            model = model_new

            # Testing
            test_accuracy, f1 = test_step(data_loader=test_loader, model=model)

            # Append training results each epoch
            train_acc_list.append(train_accuracy[0])
            train_acc_list_aug.append(train_accuracy[1])
            train_avg_loss_list.append(avg_loss)

            # Append test results each epoch
            test_acc_list.append(test_accuracy)
            test_f1s.append(f1)

            print(f'{e + 1} epoch - Train Loss {avg_loss:.4f}\tTrain accuracy {train_accuracy[0]:.4f}\tTest accuracy {test_accuracy:.4f}\tTest f1 {f1:.4f}')

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    # Save confidence numbers to CSV if specified
    if args.confcsv is not None:
        csvpath = os.path.join(args.basicpath, 'src', 'bar_info')
        if not os.path.exists(csvpath):
            os.makedirs(csvpath)
        pd.DataFrame(conf_num).to_csv(os.path.join(csvpath, args.dataset + str(args.sel_method) + args.confcsv), mode='a', header=True)

    # Save model if specified
    if args.save_model:
        save_model_and_sel_dict()

    # Plot training loss and test accuracy if specified
    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(train_avg_loss_list, test_acc_list, args, pred_precision=train_acc_list, aug_accs=train_acc_list_aug, saver=saver, save=True)

    # Return the final model and test results of the last ten epochs
    test_results_last_ten_epochs = {
        'last_ten_test_acc': test_acc_list[-10:],
        'last_ten_test_f1': test_f1s[-10:]
    }
    return model, test_results_last_ten_epochs

def train_eval_model(model, x_train, x_test, Y_train, Y_test, Y_train_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean)) # 'Y_train_clean' is used for evaluation instead of training.

    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)

    # compute noise prior
    ######################################################################################################
    # Train model

    model, test_results_last_ten_epochs = train_model(model, train_loader, test_loader, args,
                                          train_dataset=train_dataset,saver=saver)
    print('Train ended')

    ########################################## Eval ############################################

    # save test_results: test_acc(the final model), test_f1(the final model), avg_last_ten_test_acc, avg_last_ten_test_f1
    # test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
    #                               'Test', True, plt_cm=plt_cm, plt_lables=False) # evaluate_class will evaluate the final model.
    test_results = dict()
    test_results['acc'] = test_results_last_ten_epochs['last_ten_test_acc'][-1]
    test_results['f1_weighted'] = test_results_last_ten_epochs['last_ten_test_f1'][-1]
    test_results['avg_last_ten_test_acc'] = np.mean(test_results_last_ten_epochs['last_ten_test_acc'])
    test_results['avg_last_ten_test_f1'] = np.mean(test_results_last_ten_epochs['last_ten_test_f1'])

    #############################################################################################
    plt.close('all')
    torch.cuda.empty_cache()
    return test_results


def main_wrapper_MixTS(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=None):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args=args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))

    if args.use_threshold_examples is True:
      classes = classes + 1


    args.nbins = classes
    args.k_val=int(min(0.8*np.min(np.bincount(Y_train_clean)).astype(int),args.k_val))

    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)

    model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel_AE(ae=model, classifier=classifier, name='CNN').to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(s)
    saver.append_str([s])

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])
    saver.append_str(['Train: {}'.format(x_train.shape),
                      'Test: {}'.format(x_test.shape), '\r\n'])

    ######################################################################################################
    # Main loop
    if seed is None:
        seed = np.random.choice(1000, 1, replace=False)

    print('#' * shutil.get_terminal_size().columns)
    print('RANDOM SEED:{}'.format(seed).center(columns))
    print('#' * shutil.get_terminal_size().columns)

    args.seed = seed

    ni = args.ni
    saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
    # True or false
    print('+' * shutil.get_terminal_size().columns)
    print('Label noise ratio: %.3f' % ni)
    print('+' * shutil.get_terminal_size().columns)

    reset_seed_(seed)
    model = reset_model(model)

    Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
    Y_test = Y_test_clean

    test_results = train_eval_model(model, x_train, x_test, Y_train,
                                                   Y_test, Y_train_clean,
                                                   ni, args, saver_slave,
                                                   plt_embedding=args.plt_embedding,
                                                   plt_cm=args.plt_cm)
    remove_empty_dirs(saver.path)

    return test_results


def plot_train_loss_and_test_acc(avg_train_losses,test_acc_list,args,pred_precision=None,saver=None,save=False,aug_accs=None):
    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    l1 = ax.plot(avg_train_losses,'-', c='orangered', label='Training loss', linewidth=1)
    l2 = ax2.plot(test_acc_list, '-', c='blue', label='Test acc', linewidth=1)
    l3 = ax2.plot(pred_precision,'-',c='green',label='Sample_sel acc',linewidth=1)

    if len(aug_accs)>0:
        l4 = ax2.plot(aug_accs, '-', c='yellow', label='Aug acc', linewidth=1)
        lns = l1 + l2 + l3+l4
    else:
        lns = l1 + l2 + l3

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs,loc='upper right')
    # plt.legend(handles=[l1,l2],labels=["Training loss","Test acc"],loc='upper right')

    plt.axvline(args.warmup,color='g',linestyle='--')

    ax.set_xlabel('epoch',  size=18)
    ax.set_ylabel('Train loss',size=18)
    ax2.set_ylabel('Test acc',  size=18)
    plt.gcf().autofmt_xdate()
    plt.title(f'Model:new model dataset:{args.dataset}')
    plt.grid(True)

    plt.tight_layout()

    saver.save_fig(fig, name=args.dataset)

def warmup_CTW(data_loader, model, optimizer, criterion,  transform_fn=None, loss_all=None, epoch=None, args=None):
    global_step = 0
    avg_accuracy = 0.0
    avg_loss = 0.0
    model = model.train()

    # Initialize tensors to store features
    feature_original = torch.zeros((args.num_training_samples, args.embedding_size)).to(device)
    feature_aug = torch.zeros((args.num_training_samples, args.embedding_size)).to(device)

    for batch_idx, (x, y_hat, x_idx, _) in enumerate(data_loader):
        if x.shape[0] == 1:
            continue

        # Move data to device
        x, y_hat = x.to(device), y_hat.to(device)

        if transform_fn is not None:
            # Apply transformations and get augmented features
            x_aug_1, _ = transform_fn(x)
            x_aug_1 = x_aug_1.to(device).float()
            h_aug = model.encoder(x_aug_1)
            feature_aug[x_idx] = h_aug.squeeze(-1)
            out = model.classifier(h_aug.squeeze(-1))
            feature = feature_aug
        else:
            # Get original features
            h = model.encoder(x)
            feature_original[x_idx] = h.squeeze(-1)
            feature = feature_original
            out = model.classifier(h.squeeze(-1))

        # Compute loss
        model_loss = criterion(out, y_hat)
        loss_all[x_idx, epoch] = model_loss.detach().cpu().numpy()
        model_loss = model_loss.mean()

        # Backward propagation and optimization
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Accumulate loss and accuracy
        avg_loss += model_loss.item()
        acc = torch.eq(torch.argmax(out, 1), y_hat).float()
        avg_accuracy += acc.sum().cpu().numpy()
        global_step += len(y_hat)

    return (avg_accuracy / global_step, 0.0), avg_loss / global_step, model, feature

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

def interleave(l_weak, l_strong, u_weak, u_strong, takeN_l_weak, takeN_l_strong, takeN_u_weak, takeN_u_strong):
    l_weak_size = len(l_weak)
    l_strong_size = len(l_strong)
    u_weak_size = len(u_weak)
    u_strong_size = len(u_strong)

    total_size = l_weak_size + l_strong_size + u_weak_size + u_strong_size

    global_index_l_weak = []
    global_index_l_strong = []
    global_index_u_weak = []
    global_index_u_strong = []

    global_index = 0

    new_x = []
    l_weak_pointer = 0
    l_strong_pointer = 0
    u_weak_pointer = 0
    u_strong_pointer = 0

    while len(new_x) < total_size:
        ############################################################
        # takeN_l_weak
        if l_weak_pointer + takeN_l_weak <= l_weak_size:

            new_x.append(l_weak[l_weak_pointer:l_weak_pointer + takeN_l_weak])
            l_weak_pointer += takeN_l_weak
            global_index_l_weak.extend(list(np.arange(global_index, global_index + takeN_l_weak)))
            global_index += takeN_l_weak

        else:
            lastN_l_weak = l_weak_size - l_weak_pointer
            new_x.append(l_weak[l_weak_pointer:l_weak_pointer + lastN_l_weak])
            l_weak_pointer += lastN_l_weak
            global_index_l_weak.extend(list(np.arange(global_index, global_index + lastN_l_weak)))
            global_index += lastN_l_weak

        ############################################################
        # takeN_u_weak
        if u_weak_pointer + takeN_u_weak <= u_weak_size:

            new_x.append(u_weak[u_weak_pointer:u_weak_pointer + takeN_u_weak])
            u_weak_pointer += takeN_u_weak
            global_index_u_weak.extend(list(np.arange(global_index, global_index + takeN_u_weak)))
            global_index += takeN_u_weak

        else:
            lastN_u_weak = u_weak_size - u_weak_pointer
            new_x.append(u_weak[u_weak_pointer:u_weak_pointer + lastN_u_weak])
            u_weak_pointer += lastN_u_weak
            global_index_u_weak.extend(list(np.arange(global_index, global_index + lastN_u_weak)))
            global_index += lastN_u_weak

        ############################################################
        # takeN_l_strong
        if l_strong_pointer + takeN_l_strong <= l_strong_size:

            new_x.append(l_strong[l_strong_pointer:l_strong_pointer + takeN_l_strong])
            l_strong_pointer += takeN_l_strong
            global_index_l_strong.extend(list(np.arange(global_index, global_index + takeN_l_strong)))
            global_index += takeN_l_strong

        else:
            lastN_l_strong = l_strong_size - l_strong_pointer
            new_x.append(l_strong[l_strong_pointer:l_strong_pointer + lastN_l_strong])
            l_strong_pointer += lastN_l_strong
            global_index_l_strong.extend(list(np.arange(global_index, global_index + lastN_l_strong)))
            global_index += lastN_l_strong

        ############################################################
        # takeN_u_strong
        if u_strong_pointer + takeN_u_strong <= u_strong_size:

            new_x.append(u_strong[u_strong_pointer:u_strong_pointer + takeN_u_strong])
            u_strong_pointer += takeN_u_strong
            global_index_u_strong.extend(list(np.arange(global_index, global_index + takeN_u_strong)))
            global_index += takeN_u_strong

        else:
            lastN_u_strong = u_strong_size - u_strong_pointer
            new_x.append(u_strong[u_strong_pointer:u_strong_pointer + lastN_u_strong])
            u_strong_pointer += lastN_u_strong
            global_index_u_strong.extend(list(np.arange(global_index, global_index + lastN_u_strong)))
            global_index += lastN_u_strong
    new_x = torch.concat(new_x)
    assert len(new_x) == total_size

    return new_x, global_index_l_weak, global_index_l_strong, global_index_u_weak, global_index_u_strong

def train_step_CTW(data_loader, model, optimizer, criterion,  args=None,  aum_calculator=None, threshold_data_ids=None, selected_label_confidence=None, classwise_acc_confidence=None, loss_all=None, epoch=None):
    global_step = 0
    avg_accuracy = 0.0
    avg_loss = 0.0

    model.train()

    features_original = torch.zeros((args.num_training_samples, args.embedding_size)).to(device)
    features_aug = torch.zeros((args.num_training_samples, args.embedding_size)).to(device)
    noisy_mask=torch.ones(args.num_training_samples).to(device)
    for batch_idx, ((clean_x, clean_w, clean_s, clean_y, clean_idx), (noisy_x, noisy_w, noisy_s,noisy_y, noisy_idx)) in enumerate(data_loader):
        batch_size, a, b = clean_x.size()

        # Converts the index to the tensor on the device and calculates the threshold mask
        noisy_idx = noisy_idx.view(-1)
        threshold_mask = np.isin(noisy_idx, threshold_data_ids)
        threshold_mask = torch.from_numpy(threshold_mask).to(device)

        all_idx=torch.cat((clean_idx,noisy_idx),dim=0)

        # Data is placed into the device and shaped
        clean_x, clean_w, clean_s, clean_y = clean_x.to(device), clean_w.to(device).view(batch_size, a, b), clean_s.to(device).view(batch_size, a, b), clean_y.to(device)
        noisy_x, noisy_w, noisy_s,noisy_y = noisy_x.to(device).view(batch_size * args.batch_mu, a, -1), noisy_w.to(device).view(batch_size * args.batch_mu, a, -1), noisy_s.to(device).view(batch_size * args.batch_mu, a, -1), noisy_y.to(device).view(batch_size * args.batch_mu)

        # Update category confidence
        pseudo_counter_confidence = Counter(selected_label_confidence.tolist())
        for i in range(args.nbins):
            classwise_acc_confidence[i] = pseudo_counter_confidence[i] / max(pseudo_counter_confidence.values())
        classwise_acc_confidence[args.nbins - 1] = 0  # 设置最后一类的置信度为0

        # Mixed weakly and strongly supervised samples and their indexes
        takeN_l_weak, takeN_l_strong, takeN_u_weak, takeN_u_strong = 1, 1, args.batch_mu, args.batch_mu
        inputs, global_index_l_weak, global_index_l_strong, global_index_u_weak, global_index_u_strong = interleave(clean_w, clean_s, noisy_w, noisy_s, takeN_l_weak, takeN_l_strong, takeN_u_weak, takeN_u_strong)
        inputs = inputs.float()

        # Encoder output features and enhancement
        h = model.encoder(inputs)
        h_copy = h.clone()
        h_copy = torch.cat((h_copy[1:, :, :], h_copy[0, :, :].unsqueeze(0)), dim=0)
        h_copy = (1 - args.pn_strength) * h + args.pn_strength * h_copy

        # Storage enhancement features
        features_aug[clean_idx] = h[global_index_l_weak].squeeze(-1)
        features_aug[noisy_idx] = h[global_index_u_weak].squeeze(-1)

        # Classifier output
        out = model.classifier(h_copy.squeeze(-1))
        logits_clean_w = out[global_index_l_weak]
        logits_clean_s = out[global_index_l_strong]
        logits_noisy_w = out[global_index_u_weak]
        logits_noisy_s = out[global_index_u_strong]

        # Processing the raw input, obtaining the corresponding encoder features and classification outputs
        x_origin = torch.cat((clean_x, noisy_x), dim=0)
        h_origin = model.encoder(x_origin)
        features_original[torch.cat((clean_idx, noisy_idx), dim=0)] = h_origin.squeeze(-1)

        out_origin = model.classifier(h_origin.squeeze(-1))
        clean_out = out_origin[:batch_size]


        logits_w = torch.cat((logits_clean_w.detach(), logits_noisy_w.detach()), dim=0)
        logits_w_PM = torch.softmax(logits_w, dim=-1)
        # C+1 classes

        # PM
        max_logits = torch.max(logits_w_PM, dim=-1)
        mask = logits_w_PM != max_logits.values[:, None]
        partial = logits_w_PM - mask * max_logits.values[:, None]
        second_largest = torch.max(mask * logits_w_PM, dim=-1)
        second_largest = ~mask * second_largest.values[:, None]
        margins = partial - second_largest
        # APM
        ids = all_idx.detach().cpu().numpy()[np.isin(all_idx, threshold_data_ids)]
        aum_calculator.update_aums(ids, margins.cpu().numpy()[np.isin(all_idx, threshold_data_ids)], net=True)

        unsup_loss, select_confidence, pseudo_lb = consistency_loss(
            logits_noisy_s, logits_noisy_w, classwise_acc_confidence, noisy_idx, aum_calculator, clean_y, None, args.nbins, threshold_mask, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True, labels=False)

        if noisy_idx[select_confidence == 1].nelement() != 0:
            selected_label_confidence[noisy_idx[select_confidence == 1]] = pseudo_lb[select_confidence == 1]

        loss_criterion = criterion(logits_clean_w, clean_y).mean()
        mask_noisy_batch=noisy_mask[noisy_idx]

        clean_diff = torch.softmax(logits_clean_w.detach(), 1) - torch.softmax(logits_clean_s.detach(), 1)
        noisy_diff = logits_noisy_w.softmax(1).reshape(batch_size, args.batch_mu, -1).mean(dim=1) - logits_noisy_s.softmax(1).reshape(batch_size, args.batch_mu, -1).mean(dim=1)
        relative_loss = F.mse_loss(noisy_diff, clean_diff.detach(), reduction='mean')

        # ;_1
        model_loss = loss_criterion + args.L_rea * relative_loss + args.L_match * (unsup_loss*mask_noisy_batch).mean()
        noisy_mask[noisy_idx]=0
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        acc1 = torch.eq(torch.argmax(clean_out, 1), clean_y).float()
        avg_accuracy += acc1.sum().cpu().numpy()

        global_step += 1

    return (avg_accuracy / global_step, 0), avg_loss / global_step, model, features_original


def clean_selection(args, device, trainloader, feature,  aum_calculator=None, aum_threshold=None, threshold_data_ids=None):
    C = args.nbins
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.batch_size * 3, shuffle=False, num_workers=args.num_workers)
    trainNoisyLabels = torch.LongTensor(temploader.dataset.tensors[1]).to(device)
    trainNoisyLabels = torch.cat((trainNoisyLabels, trainNoisyLabels), dim=0)
    probs_norm1 = torch.zeros((len(temploader.dataset.tensors[1]), C)).to(device)

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs, targets, index, _) in enumerate(temploader):
            targets = targets.to(device)
            batchSize = inputs.size(0)
            index = index.to(device)
            similarity_matrix = F.cosine_similarity(feature[index].unsqueeze(1), feature.unsqueeze(0), dim=2)
            similarity_matrix[torch.arange(similarity_matrix.size()[0]).to(device), index] = -1
            dist = similarity_matrix

            # Top-K similar scores and corresponding indexes
            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True)

            # Replicate the labels per row to select
            candidates = trainNoisyLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            # Generate the K*batchSize one-hot encodings from neighboring labels
            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone()

            # Calculate normalized probabilities
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]
            probs_norm1[index] = probs_norm


            prob_temp = torch.max(probs_norm, dim=-1).values
            mask = probs_norm != prob_temp[:, None]
            partial = probs_norm - mask * prob_temp[:, None]
            second_largest = torch.max(mask * probs_norm, dim=-1)
            second_largest = ~mask * second_largest.values[:, None]
            margins = partial - second_largest

            # Update AUMs
            aum_calculator.update_aums(index.cpu().numpy(), margins.cpu().numpy())


    def standardization(data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)+1e-8
        return (data - mu) / sigma
    # Get AUMs and apply threshold
    crt_aums = aum_calculator.get_aums(np.arange(args.num_training_samples))
    crt_aums = crt_aums[np.arange(args.num_training_samples), trainNoisyLabels[:args.num_training_samples].cpu()]
    if args.threshold_type == 0.:
        crt_aums = torch.tensor(crt_aums).to(device)
        threshold = args.apm_threshold_weight * crt_aums.mean() + (1 - args.apm_threshold_weight) * aum_threshold
        aum_mask = crt_aums.ge(threshold).float()
    elif args.threshold_type == -1.:
        crt_aums = np.array(crt_aums)
        for cls in range(C):
            crt_aums[trainNoisyLabels[:args.num_training_samples].cpu() == cls] = standardization(
                crt_aums[trainNoisyLabels[:args.num_training_samples].cpu() == cls])

        aum_mask = torch.zeros((len(temploader.dataset.tensors[1]),)).to(device)
        top_clean_class_relative_idx = torch.topk(torch.from_numpy(crt_aums), k=int(
            crt_aums.size * (crt_aums >= crt_aums.mean()).mean()), largest=True,
                                                  sorted=False)[1]
        aum_mask[top_clean_class_relative_idx] = 1.0
    else:
        crt_aums = torch.tensor(crt_aums).to(device)
        threshold = torch.tensor(args.threshold_type).to(device)
        aum_mask = crt_aums.ge(threshold).float()
        # 0 samples may be taken
        if aum_mask.sum()==0:
            aum_mask[torch.argmax(crt_aums)]=1
        elif aum_mask.sum()==len(aum_mask):
            aum_mask[torch.argmin(crt_aums)]=0

    return aum_mask, crt_aums