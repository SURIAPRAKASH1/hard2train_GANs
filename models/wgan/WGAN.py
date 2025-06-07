import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys 
import os 
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from common.argfile import get_args

# command line args
args = get_args()

# current device
device = "cuda" if torch.cuda.is_available() else "cpu"


################################ Wassserstein Generative Adversarial Network ############################################

class Generator(nn.Module):

    def __init__(self, noise_dim: int = 100 )-> None:
        super().__init__()

        # layer 1. just a linear projection
        self.l1 = nn.Linear(noise_dim, 4 * 4 * 1024, bias = False)         

        # layer 2
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, padding = 2, output_padding= 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) # (B, 512, 8, 8)

        # layer 3
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, padding= 2, output_padding= 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # (B, 256, 16, 16)

        # layer 4
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, padding= 2, output_padding= 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) # (B, 128, 32, 32)

        # final layer
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 5, 2, padding= 2, output_padding= 1, bias = False) ,
            nn.Tanh()
        )

        # weight initialization
        self.apply(self._init_weights)

        # report parameters count
        print(f"{self.__class__.__name__} Model parameters : {(self._get_parameters_count()) / 1e+6:2f}M")

    def _get_parameters_count(self)-> int:
        """
        Returns Parameters counts in Model
        """
        t = 0
        for p in self.parameters():
            t += p.nelement()
        return t

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean = 0.0, std = 0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean = 0.0, std = 0.02)

        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean = 0.0, std = 0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean = 0.0, std = 0.02)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Args:
            x (torch.Tensor): z-noise drawn from Uniform distripution with shape (B, 100) usually

        Returns:
            out (torch.Tensor): Generated Images with shape (B, C, H, W)
        """
        out = self.tconv4(self.tconv3(self.tconv2(self.tconv1(self.l1(x).reshape(-1, 4, 4, 1024).permute(0, 3, 1, 2)))))
        return out
    

class Critic(nn.Module):

    """
    Critic Model: 
        called Critic instead of Discriminator. cause no longer classifies images as real/fake instead gives scores
    """

    def __init__(self, out_features = 1)-> None:
        super().__init__()

        # layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, padding= 2),
            nn.LeakyReLU(negative_slope= 0.2)
        )

        # layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, padding= 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope= 0.2)
        )

        # layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 2, padding= 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope= 0.2)
        )

        # layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 2, padding= 2),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(negative_slope= 0.2)
        )

        # final laeyr
        self.final = nn.Sequential(
            nn.Flatten(start_dim= 1),
            nn.Linear(512 * 4 * 4, out_features),
        )

        # weight initialization
        self.apply(self._init_weights)

        # report parameters count
        print(f"{self.__class__.__name__} Model parameters : {(self._get_parameters_count())/1e+6:2f}M")

    def _get_parameters_count(self) -> int:
        """
        Returns Parameters counts in Model
        """
        t = 0
        for p in self.parameters():
            t += p.nelement()
        return t

    def _init_weights(self, m)-> None:
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean = 0.0, std = 0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean = 0.0, std = 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean = 0.0, std = 0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean = 0.0, std = 0.02)


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Output of Generator Network or real dataset Images

        Returns:
            out (torch.Tensor): Scaler Tensor does the image is Real/Fake value b/w 0-1

        """
        out = self.final(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return out
    

##################### Initiating models ##########################
G = Generator(args.latent_dim) 
torch.complie(G) 
C = Critic()
torch.compile(C) 

print("Models are compiled !")


#####################  Optimizer ###############

# RMSprop optimizer
opt_G  = optim.RMSprop(G.parameters(), lr= args.lr)
opt_C = optim.RMSprop(C.parameters(), lr = args.lr)


####################### Datset and DataLoader #######################

# Dataset
if args.celebA:
    from datasets.celebA  import celebA_dataset
    dataset = celebA_dataset
else:
    from datasets.celebA import  mnist_dataset
    dataset = mnist_dataset

# Dataloader
dataloader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            pin_memory= True, 
                            shuffle= True,
                            drop_last= True)


###################### Optimization Loop ##########################

start = time.time()
for epoch in range(args.epochs):
    if (epoch % args.print_interval == 0 or epoch == args.epochs - 1):
        print("Epoch", epoch + 1)

    for batch, (real_imgs, _) in enumerate(dataloader, 0):
        real_imgs = real_imgs.to(device, non_blocking = True)

        for _ in range(args.n_critic):

            # ---------------------
            # A.Train Critic
            # ---------------------
            G.requires_grad_(False)
            C.requires_grad_(True)
            opt_C.zero_grad()

            noise = torch.rand(args.batch_size, args.latent_dim, device = device) * 2 - 1
            real_labels = torch.ones(args.batch_size, 1, device = device)
            fake_labels = torch.zeros(args.batch_size, 1, device = device)

            # Real Images
            pred_real = C(real_imgs)

            # Fake Images
            fake_imgs = G(noise).detach()     
            pred_fake = C(fake_imgs)

            # critic loss 
            loss_C = torch.mean(-pred_real) + torch.mean(pred_fake)

            # back-prop and update C parameters
            loss_C.backward()
            opt_C.step()

            # weight clipping for Critic to enforce 1-L continuty
            with torch.no_grad():
                for p in C.parameters():
                    p.data.clamp(-args.c, args.c)

        # B.Train Generator
        G.requires_grad_(True)
        C.requires_grad_(False)
        opt_G.zero_grad()

        noise = torch.rand((args.batch_size, args.latent_dim), device = device) * 2 - 1
        target_labels = torch.ones(args.batch_size, 1, device = device)

        generated = G(noise)

        # make critic high score to fake images
        pred = C(generated)

        # loss
        loss_G = torch.mean(-pred)

        # back-prop and update G parameters
        loss_G.backward()
        opt_G.step()

        if (epoch % 2 == 0 or epoch == args.epochs - 1) and ( batch % 200 == 0 or batch == dataloader.__len__()-1):
            print(f"{dataloader.__len__()}/{batch}: C Score: {loss_C} G Score: {loss_G}")

end = time.time()
print("training time %.2f" % ((end - start)/60),"M")