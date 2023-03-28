import copy
import logging

import gpytorch
import torch

import numpy as np



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # k(x, y) = `covar_module.raw_outputscale` * exp(- 1/2 * x * `covar_module.base_kernel.raw_lengthscale` * y)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(object):
    def __init__(self, train_x, train_y, valid_x=[], valid_y=[], state_dict=None, sigma=1e-3):
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)
        valid_x = torch.Tensor(valid_x)
        valid_y = torch.Tensor(valid_y)

        noises = torch.full_like(train_y, sigma)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=False)
        #likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = ExactGPModel(train_x, train_y, likelihood)
        self.likelihood = likelihood

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        # optimize model parameters
        #self.train_model_hyperparameters(train_x, train_y, valid_x, valid_y, sigma)

        # they are in the eval mode after tuning parameters
        self.model.eval()
        self.likelihood.eval()

        self.sigma = sigma

    def get_mean(self, x):
        # batch size = 1
        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        with torch.no_grad() and gpytorch.settings.fast_pred_var():
            x = x.unsqueeze(0)
            return self.model(x).mean.detach().numpy().flatten()[0]

    def get_variance(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        with torch.no_grad() and gpytorch.settings.fast_pred_var():
            x = x.unsqueeze(0)
            return self.model(x).variance.detach().numpy().flatten()[0]

    def find_highest_mean_and_value(self, data):
        data_and_value = [(x, self.get_mean(x)) for x in data]
        return max(data_and_value, key=lambda _: _[1])

    def update(self, query, response):
        # query: 1 x feature size
        query = query.unsqueeze(0)
        # response: 1
        response = response.unsqueeze(0)

        noises = torch.full_like(response, self.sigma)

        #self.model = self.model.get_fantasy_model(query, response)
        self.model = self.model.get_fantasy_model(query, response, noise=noises)

        self.model.eval()

    def train_model_hyperparameters(self, train_x, train_y, valid_x, valid_y, sigma):
        """
        Train GP hyper-parameters with train_x and train_y
        """
        training_iter_cands = range(0, 501, 50)
        validation_losses = []
        trained_models = []

        model = self.model
        likelihood = self.likelihood

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()}  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        logging.info('training hyper-parameters')

        validation_noises = torch.ones(valid_x.size()) * sigma
        validation_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=validation_noises,
                                                                                  learn_additional_noise=False)
        # validation_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        validation_likelihood.eval()

        # mml for computing validation loss
        valid_mml = gpytorch.mlls.ExactMarginalLogLikelihood(validation_likelihood, model)

        for iter in range(max(training_iter_cands) + 1):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()

            optimizer.step()

            if iter in training_iter_cands:
                # evaluate on the validation data
                model.eval()

                with torch.no_grad():
                    output = model(valid_x)
                    valid_loss = -valid_mml(output, valid_y)
                    validation_losses.append(valid_loss)

                    # change back to train and continue training on the training data
                    model.train()
                    likelihood.train()

                    trained_models.append(copy.deepcopy(model))

        min_valid_loss_idx = int(np.argmin(validation_losses))

        model = trained_models[min_valid_loss_idx]

        # just return likelihood as it is
        self.model = model
        self.likelihood = likelihood


def test_gp():
    train_x = torch.Tensor([[0., 0.], [1., 1.]])
    train_y = torch.Tensor([1., 0.])

    gp = GPModel(train_x, train_y)

    with torch.no_grad():
        test_x = torch.Tensor([.5, .5])

        print(gp.get_mean(test_x))

        gp.update(test_x, torch.tensor(0))

        print(gp.get_mean(test_x))
        #plot_data_points(test_x, test_y, 'x', 'y', 'test')

if __name__ == '__main__':
    test_gp()