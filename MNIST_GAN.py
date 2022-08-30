from re import S
import numpy as np
import os
import errno
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import random
import torch.optim as optim
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch import device
import torch.utils.data

class DiscriminatorNet(nn.Module):
    """
    A Neural Network that decides, wether a picture is real art or not
    """
    def __init__(self):
        super(DiscriminatorNet,self).__init__()
        nc = 32
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, nc, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128

            nn.Conv2d(nc, nc * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(nc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(1),
            # state size. (ndf*2) x 64 x 64

            
            nn.Linear(3136, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            
            # nn.Conv2d(nc * 4, nc * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(nc * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 16 x 16 

            # nn.Conv2d(nc*8, nc * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(nc * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*16) x 8 x 8
            
            # nn.Conv2d(nc * 8, nc * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(nc * 32),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*32) x 4 x 4
            
            nn.Linear(512, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.discriminator(x)


        
class GeneratorNet(nn.Module):
    """
    A Neural Network that creates synthetically generated art.
    """
    def __init__(self):
        
        super(GeneratorNet, self).__init__()
        nc = 32
        
        self.generator = nn.Sequential(
            # Block 1:input is Z, going into a convolution
            # nn.ConvTranspose2d(100, nc * 32, 5, 1, 0, bias=False),
            # nn.BatchNorm2d(nc * 32),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            
            # nn.ConvTranspose2d(nc * 32, nc * 16, 5, 2, 1, bias=False),
            # nn.BatchNorm2d(nc * 16),
            # nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            
            # nn.ConvTranspose2d(100, nc * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(nc * 8),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(100, nc*4 , 4, 2, 0, bias=False),
            nn.BatchNorm2d(nc * 4),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            
            nn.ConvTranspose2d(nc * 4, nc*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nc * 2),
            nn.ReLU(True),
            # # state size. (ngf*2) x 64 x 64

            nn.ConvTranspose2d(nc * 2, nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            
            nn.ConvTranspose2d(nc, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 256 x 256
        )

    def forward(self, input):
        return self.generator(input)
     
'''
Initialising GAN
'''  
def GAN(device, learning_rate = 0.01):
    #Initialising Generator and Discriminator
    generator = GeneratorNet().to(device)
    discriminator = DiscriminatorNet().to(device)
    
    print("Generator and Discriminator initialised")
    
    #Optimiser 
    loss = nn.BCELoss()
    
    opt_d = optim.Adam(discriminator.parameters(), lr = learning_rate)
    opt_g = optim.Adam(generator.parameters(), lr = learning_rate)
    print("Loss Function Initialising worked")
    return (generator, discriminator, loss, opt_d, opt_g)
    
    
'''
Loading Pictures
'''
def load_pictures(batch_size):
    
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))
                        ])

    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(mnist_data, batch_size = batch_size, shuffle = True)
    return dataset


def train_discriminator(discriminator, optimizer, device, loss, real_data, fake_data):
    #Reset gradients
    optimizer.zero_grad()
    
    #Train on real data
    real_data = real_data.to(device)
    real_label = torch.full((real_data.size(0),1), 1, device=device).float()
    output = discriminator(real_data.float())
    error_real = loss(output, real_label)
    error_real.backward()
    
    #Train on fake data
    fake_data = fake_data.to(device)
    fake_label = torch.full((fake_data.size(0),1), 0, device=device).float()
    output = discriminator(fake_data.float())
    error_fake = loss(output, fake_label)
    error_fake.backward()
    
    #Update optimiser
    optimizer.step()
    return error_real + error_fake
    
    
def train_generator(generator, discriminator, optimizer, device, loss, fake_data):
    #Reset gradients
    optimizer.zero_grad()
    picture_array = fake_data.to(device)
    
    #Create array of right labels, idea: change loss function so that fake pictures get identified as real paintings
    label_1 = torch.full((picture_array.size(0),1), 1, device=device).float()
    output = discriminator(picture_array)
    error = loss(output, label_1)
    error.backward()
    
    optimizer.step()
    return error


def show_image_grid(images: torch.Tensor, ncol: int):
    image_grid = make_grid(images, ncol)     # Make images into a grid
    image_grid = image_grid.permute(1, 2, 0) # Move channel to the last
    image_grid = image_grid.cpu().numpy()    # Convert into Numpy

    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

def generate_images(batch_size, generator):
    z = torch.randn(batch_size, 100, 1, 1)
    output = generator(z)
    generated_images = output
    return generated_images
    
'''
Main Function
'''
def main():
    epochs = 5
    batch_size = 64
    learning_rate = 0.001
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    '''
    Loading the GAN and Discriminator and Generator
    Loading Paintings and Pictures as a Dataloader
    '''
    generator, discriminator, loss, opt_d, opt_g = GAN(device, learning_rate)
    images = load_pictures(batch_size)
    generated_images = generate_images(batch_size, generator)
    show_image_grid(generated_images, ncol=8)
    
    
    print("starting Training")
    for epoch in range(epochs):
        for i, (image, labels) in enumerate(tqdm(images)):
            # Train discriminator 
            generated_images = generate_images(batch_size, generator)
            err_d = train_discriminator(discriminator, opt_d, device, loss, image, generated_images)
            
            # Train generator
            generated_images = generate_images(batch_size, generator)
            err_g = train_generator(generator, discriminator, opt_g, device, loss, generated_images)
            
            
            if i % 2 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, epochs, i, len(images), err_d, err_g))
            
            # #Save particular pictures before and after being edited by the generator
            # if epoch % 5 == 0:
            #     vutils.save_image(picture_arr[0],
            #         'real_samples.png',
            #         normalize=True)
            #     fake = generator(picture_arr)
            #     vutils.save_image(fake.detach()[0],
            #             '%s/fake_samples_epoch_%03d.png' % (epoch),
            #             normalize=True)
            
    print("Done")
    
main()
            
            
            
            
            
            
            
            
            

    # for i, data in enumerate(pictures, 0):
    #     # plt.imshow(data[0].permute(1,2,0))
    #     # plt.axis('off')
    #     # plt.title("Before")
    #     # plt.show()
        
    #     #generator.zero_grad()
    #     print("step 1")
    #     picture = data.to(device)
    #     print("step 2")
    #     output = generator(picture)
    #     plt.imshow(output[0].permute(1,2,0))
    #     plt.axis('off')
    #     plt.title("After")
    #     plt.show()
    #     print("step 3")
    #     print(output.size())
    # '''
    # Training the discriminator with real paintings
    # '''

    # loss_arr_paint = []
    # loss_arr_fake = []
    # D_x = []
    # D_x_fake = []
    # for data in paintings:
    #     discriminator.zero_grad()
    #     real_painting = data.to(device)
    #     label_1 = torch.full((real_painting.size(0),1), 1, device=device).float()
    #     output = discriminator(real_painting).float()
    #     real_loss = loss(output, label_1)
    #     loss_arr_paint.append(real_loss)
    #     real_loss.backward()
    #     D_x.append(output.mean().item())
    #     opt_d.step()
    # print("Done")
    
    # '''
    # Training with fake pictures
    # '''
    # print("Now to Training with fake pictures")
    # for data in pictures:
    #     discriminator.zero_grad()
    #     picture_array = data.to(device)
    #     label_0 = torch.full((picture_array.size(0),1), 0, device=device).float()
    #     output = discriminator(picture_array).float()
    #     real_loss = loss(output, label_0)
    #     loss_arr_fake.append(real_loss)
    #     real_loss.backward()
    #     D_x_fake.append(output.mean().item())
    #     opt_d.step()
    # print("Done")
    
    
    # print("Lets go to Generator Update")
    # '''
    # Update Generator
    # '''
    # for i, data in enumerate(pictures, 0):
    #     generator.zero_grad()
    #     fake_picture = data.to(device)
    #     label_1 = torch.full((fake_picture.size(0), 1), 1, device = device).float()
    #     output = discriminator(fake_picture).float()
    #     real_loss = loss(output, label_1)
    #     real_loss.backward()
    #     opt_g.step()

    #     if i%10 == 0: 
    #         vutils.save_image(real_cpu,
    #             '%s/real_samples.png' % opt.outf,
    #             normalize=True)
    #         fake = netG(fixed_noise)
    #         vutils.save_image(fake.detach(),
    #                 '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
    #                 normalize=True)

        
        
        

'''
    Code Dump:
    _________________________________________________________
    
    def display_created_paintings(self):
    Randomly chooses pictures that have been created to look like Monets painting and displays them
    pass
    
        def optimising_model(self):
    
        real_label = 1
        fake_label = 0  
        
        for epoch in range(1, self.Epochs+1):
            D_loss_list, G_loss_list = [], []
            for elem in self.real_pictures:
                #loss on real images
                self.generator.zero_grad()

                
                real_cpu = elem.to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=self.device)
                
                output = self.discriminator(real_cpu)
                D_real_loss = loss(output, label)
                D_real_loss.backward()
                D_x = output.mean().item()

                if elem[1] == 0:
                    generated_image = self.generator(elem[0])
'''