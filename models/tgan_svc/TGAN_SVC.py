import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys 
import os 
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 

from common.argfile import get_args

# command line args
args = get_args()

# current device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("current device", device)

################ Temporal Generative Adversarial Network with Singular Value Clipping - (TGAN-SVC) ######################

def singular_value_clip(weight: torch.Tensor) -> torch.Tensor:

    """
    Singular Value Decomposition clip for conv, linear weights
    """
    dim = weight.shape
    # reshape into matrix if not already MxN
    if len(dim) > 2:
        w = weight.reshape(dim[0], -1)
    u, s, v = torch.svd(w, some=True)
    s[s > 1] = 1
    return (u @ torch.diag(s) @ v.t()).view(dim)

def batchnorm_gamma_clip(module: nn.Module)-> torch.Tensor:

    """
    cliping batchnorm's learnable parameter gamma within range of  0 < gamma <= sqrt(running_var)
    """
    gamma = module.weight.data
    std = torch.sqrt(module.running_var)
    gamma[gamma > std] = std[gamma > std]
    gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
    return gamma


class TemporalGenerator(nn.Module):

    """
    TemporalGenerator:
        For given noise generates latent temporal (time) features
    """
    def __init__(self):
        super().__init__()

        # Create a sequential model to turn one vector into 16
        self.model = nn.Sequential(
            nn.ConvTranspose1d(100, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 100, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # initialize weights according to paper
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def forward(self, x):
        # reshape x so that it can have convolutions done
        x = x.view(-1, 100, 1)
        # apply the model and flip the
        x = self.model(x).transpose(1, 2)
        return x


class VideoGenerator(nn.Module):
    """
    VideoGenerator:
        Generated Images from given latent temporal features
    """

    def __init__(self):
        super().__init__()

        # instantiate the temporal generator
        self.temp = TemporalGenerator()

        # create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # define the image generator
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # initialize weights according to the paper
        self.fast.apply(self.init_weights)
        self.slow.apply(self.init_weights)
        self.model.apply(self.init_weights)

        print(f"{self.__class__.__name__} parameters: {(self._get_parameters_count() / 1e+6):.2f} M")

    def _get_parameters_count(self)-> int:
        """
        Returns total parameters in model
        """
        t = 0
        for p in self.parameters():
            t += p.nelement()
        return t

    def device(self)-> torch.DeviceObjType:
        return next(self.parameters()).device

    def init_weights(self, m)-> None:
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Takes noise Generates video

        Args:
            x (torch.Tensor) : prior lantent noise vector randomly drawn. Shape (B, 100)

        Returns:
            out (torch.Tensor): Generated Videos with Shape (B, T, C, H, W) -> (batch_size, (time)frames, channels, height, width)

        """

        # pass our latent vector through the temporal generator and reshape
        z_fast = self.temp(x).contiguous()       # (B, T, 100)
        z_fast = z_fast.view(-1, 100)            # (B*T, 100)

        # transform (just a linear projection) the content and temporal vectors
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)                      # (B*T, C, H, W)
        z_slow = self.slow(x).view(-1, 256, 4, 4).unsqueeze(1)              # (B, 1, C, H, W)

        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow]*16, dim=1).view(-1, 256, 4, 4)          # (B*T, C, H, W)

        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1)                              # (B*T, C + C, H, W)

        # transform into image frames
        out = self.model(z)

        return out.view(-1, 16, 3, 64, 64).transpose(1, 2)             # reshape and transpose (B, C, T, H, W)


class VideoCritic(nn.Module):
    """

    VideoCritic:
        VideoCritic scores weather the video came from real/fake distribution

    """

    def __init__(self):
        super().__init__()

        self.model3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )

        self.conv2d = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

        # initialize weights according to paper
        self.model3d.apply(self.init_weights)
        self.init_weights(self.conv2d)

        # report parameters
        print(f"{self.__class__.__name__} parameters: {(self._get_parameters_count() / 1e+6):.2f} M")

    def _get_parameters_count(self)-> int:
        """
        Returns total parameters in model
        """
        t = 0
        for p in self.parameters():
            t += p.nelement()
        return t

    def device(self)-> torch.DeviceObjType:
        return next(self.parameters()).device

    def init_weights(self, m):
        if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, gain=2**0.5)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Videos from Real/Fake dist with shape (B, C, T, H, W)

        Returns:
            h (torch.Tensor): scores with shape (B, ) weather a vedio is from real/fake distribution

        """

        h = self.model3d(x)
        # turn a tensor of R^NxCxTxHxW into R^NxCxHxW
        h = torch.reshape(h, (args.batch_size, 512, 4, 4))        # (B, C, H, W)
        h = self.conv2d(h).view(-1)                  # (B, 1, 1, 1) -> (B)
        return h


##################### Initiating models ##########################
print("Initiating Models ......")

VG = VideoGenerator().to(device)
torch.compile(VG)
VC = VideoCritic().to(device)
torch.compile(VC)

print("Models are compiled !")


#####################  Optimizer #####################################

# Video Generator optimizer
ropt_VG = optim.RMSprop(VG.parameters(), lr = args.lr)
# Video Critic optimizer
ropt_VC = optim.RMSprop(VC.parameters(), lr = args.lr)


####################### Datset and DataLoader #######################

from gan_datasets.moving_mnist import get_mmnist_dataset
mmnist_root: str =  "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
dataset = get_mmnist_dataset(mmnist_root)
# Dataloader
dataloader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            pin_memory= True, 
                            shuffle= True,
                            drop_last= True,
                            num_workers = os.cpu_count() - 1)


###################### Optimization Loop ##########################

start = time.time()
for epoch in range(args.epochs):
    if (epoch % args.print_interval == 0 or epoch == args.epochs - 1):
        print("Epoch", epoch + 1)

    for current_batch, real_videos in enumerate(dataloader, 0): # videos with shape (B, T, C, H, W)
        real_vids = real_videos.to(device, non_blocking = True)

        for _ in range(args.n_critic):

            # -----------------
            # A.Training Critic
            # ------------------
            VG.requires_grad_(False)
            VC.requires_grad_(True)
            ropt_VC.zero_grad()

            # z-noise
            noise = torch.rand((args.batch_size, args.latent_dim), device = device) * 2 - 1

            # Real Image Batch
            pred_real = VC(real_vids)

            # Fake Image Batch
            fake_vids = VG(noise).detach()     # make sure we don't train VG when traning VC
            pred_fake = VC(fake_vids)

            # critic loss
            loss_VC = -pred_real.mean() + pred_fake.mean()

            # Back-prop & update Critic parameters
            loss_VC.backward()
            ropt_VC.step()

        # ---------------------
        # B.Train VideoGenerator
        # ----------------------
        VG.requires_grad_(True)
        VC.requires_grad_(False)
        ropt_VG.zero_grad()

        noise = torch.rand((args.batch_size, args.latent_dim), device = device) * 2 - 1
        generated = VG(noise)
        pred = VC(generated)
        loss_VG = -1 * pred.mean()

        # back-prop and update G parameters
        loss_VG.backward()
        ropt_VG.step()

        # 1-L enforce for only Discriminator (i mean Critic)
        if current_batch + 1 % 5 == 0:
            for module in list(VC.model3d.children()) + [VC.conv2d]:
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                    module.weight.data = singular_value_clip(module.weight)
                elif isinstance(module, nn.BatchNorm3d):
                    module.weight.data = batchnorm_gamma_clip(module)


        if current_batch + 1 % args.ckp_interval == 0:
            torch.save(VG.state_dict(), "checkpoints/tgan_VGckp.pt")
            torch.save(VC.state_dict(), "checkpoints/tgan_VCckp.pt")
            print("checkpoints are saved ...!")

        if (epoch % args.print_interval == 0 or epoch == args.epochs - 1) and ( current_batch % 50 == 0 or current_batch == dataloader.__len__()-1):
            print(f"{dataloader.__len__()}/{current_batch}, VC Score: {loss_VC} , VG Score: {loss_VG}")

end = time.time()
print("training time %.2f" % ((end - start)/60),"M")
