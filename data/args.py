import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='DHS-IMR-MaternalEd')
    parser.add_argument(
        '--smokescreen', default=False, help='whether to test training process with reduced computational requirements')
    parser.add_argument(
        '--exp_name', default='test', help='name of experiment, must be convertable to a directory name')
    parser.add_argument(
        '--load', default=False, help='whether to load a model')
    parser.add_argument(
        '--load_dir', default='', help="name of experiment directory within 'runs'")
    parser.add_argument(
        '--eval_only', default=False, help='whether to evaluate model only (prevents training)')
    parser.add_argument(
        '--verbose', default=0, help='level of verbosity in outputs. 0 for just train completion percentage, 1 for periodic loss outputs')
    parser.add_argument(
        '--eval_test', default=False, help='whether to perform final evaluation on the trained/loaded model with the test set, otherwise use val set'
    )
    # parser.add_argument(
    #     '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    # parser.add_argument(
    #     '--eps',
    #     type=float,
    #     default=1e-5,
    #     help='RMSprop optimizer epsilon (default: 1e-5)')
    # parser.add_argument(
    #     '--alpha',
    #     type=float,
    #     default=0.99,
    #     help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--seed', type=int, default=7, help='random seed (default: 7)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=4,
        help='how many training CPU processes to use (default: 4)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
