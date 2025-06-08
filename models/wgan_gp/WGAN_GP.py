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
print("current device -->", device)

######################### Wassserstein Generative Adversarial Network with Gradinet Penalty ##########################

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
    

################################################# Gradient Penalty ###########################################################

def compute_gradient_penalty(C: nn.Module,
                             real_imgs: torch.Tensor,
                             fake_imgs: torch.Tensor,
                             lamda: int) -> torch.Tensor:
    """
    Computes gradient penalty w.r.t interpolated image. lamda(l2_norm(x_hat_grad(C(x_hat)) - 1)**2)

    Args:
        C (nn.Module): Critic model that takes Fake/Interpolated Image and gives scores
        real_imgs (torch.Tensor): Real Images from dataset
        fake_imgs (torch.Tensor): Fake Images from Generator model
        lamda (torch.Tensor): Co-efficient term for penalty

    Returns:
        gradient_penalty (torch.Tensor): Scalar gradient penalty

    """

    # random number drawn from uniform distripution
    eps = torch.rand((args.batch_size, 1, 1, 1), device = real_imgs.device)
    # interpolated images
    x_hat = eps * real_imgs + (1 - eps) * fake_imgs
    x_hat.requires_grad_(True)
    # Discriminator's (i mean Critic) score on interpolated images
    critic_score = C(x_hat)

    # compute gradients w.r.t x_hat
    gradients = torch.autograd.grad(
        critic_score,
        x_hat,
        torch.ones_like(critic_score),
        retain_graph = True,
        create_graph = True
    )[0] # (batch, channels, height, weight)

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lamda * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
    return gradient_penalty

##################### Initiating models ##########################

G = Generator(args.latent_dim) 
torch.compile(G) 
C = Critic()
torch.compile(C) 

print("Models are compiled !")


#####################  Optimizer ###############

# Adam optimizer for Generator and Critic
opt_G = optim.AdamW(G.parameters(), lr = args.lr, betas= (0.0, 0.9))
opt_C = optim.AdamW(C.parameters(), lr = args.lr, betas = (0.0, 0.9))


####################### Datset and DataLoader #######################

# Dataset
if args.celebA:
    from gan_datasets.celebA import get_celebA_dataset
    celebA_root: str = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing"
    dataset = get_celebA_dataset(celebA_root)
else:
    from gan_datasets.mnist import get_mnist_dataset
    mnist_root: str = "<http://yann.lecun.com/exdb/mnist/>"
    dataset = get_mnist_dataset(mnist_root)

# Dataloader
os.makedirs("checkpoints", exist_ok = True)
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

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(C, real_imgs, fake_imgs, args.lamda)

            # critic loss 
            loss_C = torch.mean(-pred_real) + torch.mean(pred_fake) + gradient_penalty

            # back-prop and update C parameters
            loss_C.backward()
            opt_C.step()

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

        if batch % args.ckp_interval == 0:
            torch.save(G.state_dict(), "checkpoints/wgangp_Gckp.pt")
            torch.save(C.state_dict(), "checkpoints/wgangp_Dckp.pt")
            print("checkpoints are saved ...!")

        if (epoch % args.print_interval == 0 or epoch == args.epochs - 1) and ( batch % 200 == 0 or batch == dataloader.__len__()-1):
            print(f"{dataloader.__len__()}/{batch}, C Score: {loss_C} , G Score: {loss_G}")

end = time.time()
print("training time %.2f" % ((end - start)/60),"M")

