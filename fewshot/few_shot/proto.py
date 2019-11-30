import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable

from few_shot.utils import pairwise_distances
from few_shot.losses import IdentityTransformLoss
from few_shot.stn import STNv0

stnidentityloss = IdentityTransformLoss()

def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool,
                      stnmodel = None,
                      stnoptim = None,
                      args = None,):

    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
        if stnmodel:
            stnmodel.train()
            stnoptim.zero_grad()
    else:
        model.eval()
        if stnmodel:
            stnmodel.eval()

    # If there is an STN, then modify some of the samples
    theta = None
    info = None
    if stnmodel:
        if args.targetonly:
            supnum = n_shot*k_way
            xsup, thetasup, info = stnmodel(x[:supnum], 1)
            xtar, thetatar, info = stnmodel(x[supnum:], 0)
            x = torch.cat([xsup, xtar], 0)
            theta = torch.cat([thetasup, thetatar], 0)
        else:
            x, theta, info = stnmodel(x)

    # Embed all samples
    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance, model)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Calculate the stn loss
    if stnmodel and train:
        #print(loss, stnidentityloss(theta))
        loss = -loss + args.stn_reg_coeff * stnidentityloss(theta)
        loss.backward()
        #for p in stnmodel.parameters():
            #print(p.grad)
        stnoptim.step()

        # Reset optimizers
        optimiser.zero_grad()

        # Prediction probabilities are softmax over distances
        # Embed all samples
        embeddings = model(x.detach())

        # Samples are ordered by the NShotWrapper class as follows:
        # k lots of n support samples from a particular class
        # k lots of q query samples from those classes
        support = embeddings[:n_shot*k_way]
        queries = embeddings[n_shot*k_way:]
        prototypes = compute_prototypes(support, k_way, n_shot)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(queries, prototypes, distance)

        # Calculate log p_{phi} (y = k | x)
        log_p_y = (-distances).log_softmax(dim=1)
        loss = loss_fn(log_p_y, y)

    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes
