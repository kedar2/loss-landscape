import torch
from models import MLP
from metrics import jacobian
from train import train_model
from data_generation import RandomPolynomialMapping
import argparse
import utils

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--degree', type=int, default=1)
    parser.add_argument('--num_data', type=int, default=500)
    args = parser.parse_args()


    model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_hidden_layers=args.num_hidden_layers)
    poly = RandomPolynomialMapping(-1, 1, args.degree)
    X, y = poly.generate_gaussian_data(args.input_dim, args.num_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    model, logged_metrics = train_model(model, optimizer, loss_fn, X, y, args.num_epochs)
    utils.export_dict_to_csv(logged_metrics, 'results/twolayer.csv')
    utils.plot_dict_values(logged_metrics)