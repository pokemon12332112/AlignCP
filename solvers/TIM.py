import torch
import random
import tqdm

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def compute_codes(Adapter, features, observed_marginal=False, labels_count=None):

    if observed_marginal:
        label_dist = labels_count / np.sum(labels_count)
        r = torch.tensor(label_dist).to(device)
        alpha = 1.0
    else:
        r = torch.ones(Adapter.adapter.prototypes.shape[0]).to(device) / Adapter.adapter.prototypes.shape[0]
        alpha = 1.0

    z = tim(Adapter, features, marginal=r, disp=False, kl=observed_marginal is True, alpha=alpha)
    torch.cuda.empty_cache()

    return z


def tim(Adapter, features, marginal, base_lr=0.01, iterations=100, bs=50000, disp=True, alpha=0.1, kl=True):

    n, K = features.shape

    bs = min(bs, n)

    features = features.to(device)


    epochs = int(iterations / (n/bs))


    optim = torch.optim.Adam(params=Adapter.parameters(), lr=base_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=iterations)

    for i_epoch in range(epochs):

        indexes = np.arange(0, n)
        random.shuffle(indexes) 
        tracking_loss = 0.0

        for i_step in range(max(1, n // bs)):

            init = int(i_step * bs)
            end = int((1 + i_step) * bs)

            x = features[indexes[init:end], :].to(device).to(torch.float32)
            logits = Adapter.forward(x)
            pik = torch.softmax(logits, -1)
            pk = torch.mean(pik, dim=0)

            H_yx = - torch.mean(torch.sum(pik * torch.log(pik + 1e-3), -1))

            if kl:
                Lmarg = torch.sum(marginal.clone() * torch.log((marginal.clone() / pk) + 1e-3))
            else:
                Lmarg = torch.sum(pk * torch.log(pk + 1e-3))

            loss = Lmarg + alpha * H_yx

            loss.backward()
            optim.step()
            optim.zero_grad()

            tracking_loss += loss.item() / (max(1, n // bs))

            scheduler.step()

            if disp:
                print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                    i_epoch=i_epoch + 1, epochs=epochs, i_step=i_step + 1, steps=int(n // bs),
                    loss=round(loss.item(), 4)), end="\r")

        if disp:
            print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                i_epoch=i_epoch + 1, epochs=epochs, i_step=int(n // bs),
                steps=int(n // bs), loss=round(tracking_loss, 4)), end="\n")

    Adapter.eval()

    with torch.no_grad():
        z = torch.softmax(Adapter(features).to(device), -1).cpu()

    return z