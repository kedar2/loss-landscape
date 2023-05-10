def run(input_dim: int=10,
        hidden_dim: int=64,
        output_dim: int=1,
        num_hidden_layers: int=1,
        lr: float=0.01,
        num_epochs: int=1000,
        degree: int=2,
        num_data: int=500) -> dict:
    """
    Train a two-layer MLP on random polynomial data and track gradient information.
    """
    import torch
    from models import MLP
    from train import train_model
    from data_generation import RandomPolynomialMapping

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_hidden_layers=num_hidden_layers)
    poly = RandomPolynomialMapping(-1, 1, degree)
    X, y = poly.generate_random_data(input_dim, num_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model, logged_metrics = train_model(model, optimizer, loss_fn, X, y, num_epochs)
    return logged_metrics

def main():
    import sys
    sys.path.append('..')

    import argparse
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension of network')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of network')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension of network')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of hidden layers in network')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--degree', type=int, default=2, help='Degree of polynomial mapping')
    parser.add_argument('--num_data', type=int, default=500, help='Number of data points to generate')
    args = parser.parse_args()
    logged_metrics = run(**vars(args))
    utils.export_dict_to_csv(logged_metrics, '../results/twolayer.csv')
    utils.plot_dict_values(logged_metrics, directory='../results/')

if __name__ == '__main__':
    main()