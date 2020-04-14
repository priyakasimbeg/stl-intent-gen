import numpy as np
import torch
from utils import prob_utils as ut
from torch import nn, optim

def train(model, train_loader, labeled_subset, devise, tqdm, writer,
          iter_max = np.inf, iter_save=np.inf, model_name='model',
          y_status='none', reinitialize=False, lr=1e-3):

    if reinitialize:
        model.apply(ut.reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    i = 0

    with tqdm(total=iter_max) as progress_bar:
        while True:
            for batch_idx, (x, y) in enumerate(train_loader): # Make parallel
                i += 1
                optimizer.zero_grad()

                # compute loss
                loss, summaries = model.loss(y, x)

                # optimization step
                optimizer.step()

                # progress bar
                progress_bar.set_post_fix(
                    loss='{:.2e}'.format(loss)
                )

                # # Log summaries
                # if i % 50  == 0: ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save ==0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return