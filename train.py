import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dcgan import Discriminator, Generator, initialize_weights

# Hyperparameters as per paper specifications
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 2e-4
batch_size = 128
img_size = 64
img_channels = 1
z_dim = 100
features_gen = 64
features_disc = 64
n_epochs = 5

# Dataset preparation and Model initialization
transforms = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor(),
                                 transforms.Normalize(
                                     [0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
                                 ])

dataset = MNIST(root='/dataset', train=True, download=True, transform=transforms)
loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
gen = Generator(z_dim, img_channels, features_gen).to(device)
disc = Discriminator(img_channels, features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

# Optimizer and Criterion
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=[0.5, 0.999]) # as per paper specifications
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=[0.5, 0.999])
criterion = nn.BCELoss()

fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)

# Tensorboard Initialization
writer_real = SummaryWriter(f"runs/real")
writer_fake = SummaryWriter(f"runs/fake")

# Adding Model Graphs
# For Discriminator
dummy_input_disc = torch.randn(batch_size, img_channels, img_size, img_size).to(device)
writer_real.add_graph(disc, input_to_model=dummy_input_disc)

# For Generator
dummy_input_gen = torch.randn(batch_size, z_dim, 1, 1).to(device)
writer_fake.add_graph(gen, input_to_model=dummy_input_gen)

step = 0

# Training Loop
for epoch in range(n_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # Train Discriminator
        # For Discriminator: max log(D(x)) + log(1-D(G(Z)))
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        disc_fake = disc(fake).reshape(-1)
        loss_discR = criterion(disc_real, torch.ones_like(disc_real))
        loss_discF = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = 0.5*(loss_discF + loss_discR)

        # Backpropagation
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        # For Generator: min log(1-D(G(z))) <--> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(disc_fake, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}]: lossD: {loss_disc: .4f}\tlossG: {loss_gen: .4f}")

        with torch.no_grad():
            fake = gen(fixed_noise)
            # Take out (upto) 32 images
            img_grid_real = torchvision.utils.make_grid(
                real[:32], normalize=True
            )
            img_grid_fake = torchvision.utils.make_grid(
                fake[:32], normalize=True
            )
            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        step +=1



