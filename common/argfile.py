import argparse

def get_args():
    """
        Parser for command line arguments
    """

    parser =  argparse.ArgumentParser()
    parser.add_argument("--epochs", type= int, default= 1, help= "total Epochs to train model")
    parser.add_argument("--lr", type = float, default = 0.0002, help= "learning rate to update parameters")
    parser.add_argument("--batch_size", type = int, default= 64, help= "batch_size for dataset")
    parser.add_argument("--latent_dim", type = int, default = 100, help = "Noise vector dimention")
    parser.add_argument("--height", type = int, default= 64, help = "height of the image")
    parser.add_argument("--width", type = int, default= 64, help = "width of the image")
    parser.add_argument("--channels", type=int, default= 3, help= 'number of channels in image')
    parser.add_argument("-cA" ,"--celebA", action = "store_true", help = "if flag given model will train on celebA dataset if not mnist dataset" )
    parser.add_argument("--print_interval", type= int, default= 1, help= "when training how often we want to see loss")
    parser.add_argument("--n_critic", type= int, default= 5, help= "steps for training critic model")
    parser.add_argument("--c", type= int, default= 0.01, help= "weight cliping range for wgan")
    parser.add_argument("--lamda", type= int, default= 10, help= "co-efficient term in gradient penaly for wgan-gp")
    parser.add_argument("--checkpoint_interval", type = int, default= 200, help= "Saving models checkpoint for given interval")

    return parser.parse_args()