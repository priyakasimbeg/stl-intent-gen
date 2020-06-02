import numpy as np
import torch
from utils import prob_utils as ut
from torch import nn, optim

def train(model, train_loader, device, tqdm, writer,
          iter_max = np.inf, iter_save=np.inf,
          reinitialize=False, lr=1e-3, pstl=False):

    if reinitialize:
        model.apply(ut.reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    i = 0

    if not pstl:
        with tqdm(total=iter_max) as progress_bar:
            while True:
                for batch_idx, (x, y) in enumerate(train_loader): # Todo Make parallel
                    i += 1
                    optimizer.zero_grad()

                    # compute loss
                    x.to(device)
                    y.to(device)
                    loss, summaries = model.loss(y, x)
                    rec = summaries['gen/rec']
                    kl = summaries['gen/kl_z']

                    # optimization step
                    loss.backward()
                    optimizer.step()

                    # progress bar
                    progress_bar.update()
                    progress_bar.set_postfix(
                        loss='{:.2e} - {:.2e} - {:.2e}'.format(loss, rec, kl)
                    )

                    #  Log summaries
                    if i % 50 == 0:
                        ut.log_summaries(writer, summaries, i)

                    # Save model
                    if i % iter_save ==0:
                        ut.save_model_by_name(model, i)

                    if i == iter_max:
                        return
    else:
        with tqdm(total=iter_max) as progress_bar:
            while True:
                for batch_idx, (x, y, k) in enumerate(train_loader):  # Todo Make parallel
                    i += 1
                    optimizer.zero_grad()

                    # compute loss
                    x.to(device)
                    y.to(device)
                    k.to(device)
                    loss, summaries = model.loss(y, x, k)
                    rec = summaries['gen/rec']
                    kl = summaries['gen/kl_z']
                    rob = summaries['gen/robustness']


                    # optimization step
                    loss.backward()
                    optimizer.step()

                    # progress bar
                    progress_bar.update()
                    progress_bar.set_postfix(
                        loss='{:.2e} - {:.2e} - {:.2e}'.format(loss, rec, kl)
                    )

                    #  Log summaries
                    if i % 50 == 0:
                        ut.log_summaries(writer, summaries, i)

                    # Save model
                    if i % iter_save == 0:
                        ut.save_model_by_name(model, i)

                    if i == iter_max:
                        return