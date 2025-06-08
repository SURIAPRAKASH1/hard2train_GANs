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

######################### Deep Convolutional Generative Adversarial Network - DCGAN #####################################

class Generator(nn.Module):

    """
    Generator Model: Uses Conv nets instead of FCN for generating images
    """

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

class Discriminator(nn.Module):

    """
    Discriminator Model: Uses Conv nets instead of FCN for classifies image as Fake/Real
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
print("Models Initiating ...")

G = Generator(args.latent_dim) 
torch.compile(G) 
D = Discriminator() 
torch.compile(D) 

print("Models are compiled !")


##################### Initiating Loss and optimizer ###############

# Binary Cross Entropy Loss
criterion = nn.BCEWithLogitsLoss().to(device)

# Generator optimizer
opt_G = optim.AdamW(G.parameters(), lr = args.lr, betas= (0.5, 0.5))
# Discriminator optimier
opt_D = optim.AdamW(D.parameters(), lr = args.lr, weight_decay = 1e-1, betas= (0.5, 0.5))


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
dataloader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            pin_memory= True, 
                            shuffle= True,
                            drop_last= True)


###################### Optimization Loop ##########################
print("Training Model ........")
start = time.time()
for epoch in range(args.epochs):
    if (epoch % args.print_interval == 0 or epoch == args.epochs - 1):
        print("Epoch", epoch + 1)

    for batch, (real_imgs, _) in enumerate(dataloader, 0):
        real_imgs = real_imgs.to(device, non_blocking = True)

        for _ in range(1):

            # ---------------------
            # A.Train Discriminator
            # ---------------------
            G.requires_grad_(False)
            D.requires_grad_(True)
            opt_D.zero_grad()

            noise = torch.rand(args.batch_size, args.latent_dim, device = device) * 2 - 1
            real_labels = torch.ones(args.batch_size, 1, device = device)
            fake_labels = torch.zeros(args.batch_size, 1, device = device)

            with torch.autocast(device_type= device, dtype= torch.bfloat16):

                # Real Images
                pred_real = D(real_imgs)
                loss_real = criterion(pred_real, real_labels)

                # Fake Images
                fake_imgs = G(noise).detach()     # make sure we don't train G when traning D
                pred_fake = D(fake_imgs)
                loss_fake = criterion(pred_fake, fake_labels)

            # back-prop and update D parameters
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

        # B.Train Generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        opt_G.zero_grad()

        noise = torch.rand((args.batch_size, args.latent_dim), device = device) * 2 - 1
        target_labels = torch.ones(args.batch_size, 1, device = device)

        with torch.autocast(device_type= device, dtype= torch.bfloat16):
            generated = G(noise)
            # fool the D to think it's reciving real images
            pred = D(generated)
            loss_G = criterion(pred, target_labels)

        # back-prop and update G parameters
        loss_G.backward()
        opt_G.step()

        if batch + 1 % args.ckp_interval == 0:
            torch.save(G.state_dict(), "checkpoints/dcgan_Gckp.pt")
            torch.save(D.state_dict(), "checkpoints/dcgan_Dckp.pt")
            print("checkpoints are saved ...!")

        if (epoch % args.print_interval == 0 or epoch == args.epochs - 1) and ( batch % 200 == 0 or batch == dataloader.__len__() - 1):
            print(f"{dataloader.__len__()}/{batch}: D: loss_real {loss_real}, loss_fake {loss_fake} G: loss {loss_G}")

end = time.time()
print("training time %.2f" % ((end - start)/60),"M")
