import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def adapt(features, labels, model):


    z, model = ss_text_solver(features, labels, model)

    return z, model


def ss_text_solver(features, labels, model):

    N = labels.shape[0]

    with torch.no_grad():

        affinity_labeled = torch.nn.functional.one_hot(labels).float()

        tau = (1 / model.adapter.logit_scale.exp().item()) 
        vision_mu = torch.einsum('ij,ik->jk', affinity_labeled, features) / tau

        text_mu = model.adapter.prototypes.data.clone().to("cpu")

        lambda_text = torch.tensor((1 / (N)))

        lambda_text = lambda_text.clamp(min=1e-3)

        new_mu = (1/N) * (1/lambda_text) * vision_mu + text_mu

    model.adapter.prototypes.data = new_mu.to(device)

    with torch.no_grad():
        z = torch.softmax(model(features.to(device)), -1)

    return z, model
