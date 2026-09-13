import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools



# this has to be the same architecture that we use in the HuSCFGAN, but here the embedding layers are left out 
# because they don't contribute to any flops, they are simply lookup layers, that's why we don't write them here
# if you need to try any other architecture, you need to change the architecture here as well
class Generator(nn.Module):
    def __init__(self, z_dim=100,num_classes=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            # Fully connected layer to convert z_dim to a 7x7 feature map
            nn.Linear(z_dim+num_classes, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
            
        )
        self.deconv = nn.Sequential(
            # First transpose convolution (upsampling from 7x7x256 to 14x14x128)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (Batch, 128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d(128, 128, 3, 1, 1),  # Kernel size 3, stride 1, padding 1 to preserve shape
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Second transpose convolution (upsampling to 28x28x64)
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (Batch, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Third transpose convolution (output to 28x28x1)
            nn.ConvTranspose2d(64, 1, 3, 1, 1),  # (Batch, 1, 28, 28)
            nn.Tanh()  # Output between [-1, 1]
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 256, 7, 7)
        return self.deconv(z)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # First convolution (from 1x28x28 to 64x14x14)
            nn.Conv2d(2, 64, 4, 2, 1),  # (Batch, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Second convolution (downsampling to 7x7x128)
            nn.Conv2d(64, 128, 4, 2, 1),  # (Batch, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv2d(128, 128, 3, 1, 1),  # Kernel size 3, stride 1, padding 1 to preserve shape
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third convolution (downsampling further)
            nn.Conv2d(128, 256, 4, 2, 1),  # (Batch, 256, 3, 3)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Flattening and final dense layer to classify as real or fake
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()  # Output: Probability of being real or fake
        )

    def forward(self, x):
        return self.conv(x)


# Flops calculation according to each layer
def compute_flops(layer, input_tensor,order):
    """Computes the FLOPs for a given layer."""
    forward_flops = 0
    backward_flops=0
    if isinstance(layer, nn.Conv2d):
        # FLOPs for Convolution: 2 * (kernel_height * kernel_width * input_channels * output_height * output_width * output_channels)
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops = 2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_shape[2] * output_shape[3] + layer.out_channels*output_shape[2]*output_shape[3]
        if order==0:
            backward_flops=2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_shape[2] * output_shape[3] + layer.out_channels*output_shape[2]*output_shape[3]
        else:
            backward_flops=4 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_shape[2] * output_shape[3] + layer.out_channels*output_shape[2]*output_shape[3]
    if isinstance(layer, nn.ConvTranspose2d):
        # FLOPs for Convolution: 2 * (kernel_height * kernel_width * input_channels * output_height * output_width * output_channels)
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops = 2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_shape[2] * output_shape[3] + layer.out_channels*output_shape[2]*output_shape[3]
        if order==0:
            backward_flops=2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_shape[2] * output_shape[3] + layer.out_channels*output_shape[2]*output_shape[3]
        else:
            backward_flops=4 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_shape[2] * output_shape[3] + layer.out_channels*output_shape[2]*output_shape[3]
    
    
    elif isinstance(layer, nn.Linear):
        # FLOPs for Linear: 2 * input_features * output_features
        forward_flops = 2 * layer.in_features * layer.out_features + layer.out_features
        if order ==0:
            backward_flops= 2 * layer.in_features * layer.out_features + layer.out_features
        else:
            backward_flops= 4 * layer.in_features * layer.out_features + layer.out_features
    
    elif isinstance(layer, nn.BatchNorm2d):
        # FLOPs for BatchNorm: 2 * 2 * output_channels * H * W
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops =  output_shape[1 ]* (8*output_shape[2] * output_shape[3])
        backward_flops =  output_shape[1 ]* (11*output_shape[2] * output_shape[3])
        
    elif isinstance(layer, nn.BatchNorm1d):
    # FLOPs for BatchNorm: 2 * 2 * output_channels * H * W
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops =  output_shape[1]* 8
        backward_flops=output_shape[1]*11
        
    
    elif isinstance(layer, nn.ReLU):
        # ReLU operation (elementwise): 1 operation per element
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops=1
        backward_flops=2
        for i in range(len(output_shape)):
            if i!=0:
                    forward_flops *= output_shape[i]
                    backward_flops *= output_shape[i]
        
    elif isinstance(layer, nn.LeakyReLU):
        # ReLU operation (elementwise): 1 operation per element
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops=2
        backward_flops=3
        for i in range(len(output_shape)):
            if i!=0:
                    forward_flops *= output_shape[i]
                    backward_flops *= output_shape[i]
    
    elif isinstance(layer, nn.Tanh):
        # ReLU operation (elementwise): 1 operation per element
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops=5
        backward_flops=3
        for i in range(len(output_shape)):
            if i!=0:
                    forward_flops *= output_shape[i]
                    backward_flops *= output_shape[i]
                    
    elif isinstance(layer, nn.Sigmoid):
        # ReLU operation (elementwise): 1 operation per element
        output_tensor = layer(input_tensor)
        output_shape = output_tensor.shape
        forward_flops=3
        backward_flops=2
        for i in range(len(output_shape)):
            if i!=0:
                    forward_flops *= output_shape[i]
                    backward_flops *= output_shape[i]
    return (input_tensor.shape[0]* forward_flops,input_tensor.shape[0]* backward_flops)


def calculate_flops(model, input_tensor):
    """Calculates FLOPs for every layer in the model and returns a dictionary with layer names as keys."""
    forward_flops_dict = {}
    backward_flops_dict = {}
    input_data = input_tensor
    previous_layer=None
    i=0
    for name, layer in model.named_children():
        # print(previous_layer)
        if  isinstance(layer,nn.Sequential):
            for name2,layer2 in layer.named_children():
                if  isinstance(previous_layer,nn.Linear) and (isinstance(layer2,nn.Conv2d) or isinstance(layer2,nn.ConvTranspose2d)):
                    input_data=input_data.view(input_data.shape[0],layer2.in_channels,int(np.sqrt(input_data.shape[1]/layer2.in_channels)),-1)

                layer_flops = compute_flops(layer2, input_data,i) 
                i=i+1
                forward_flops_dict[name+"."+name2] = layer_flops[0]
                backward_flops_dict[name+"."+name2] = layer_flops[1]
                # Pass the input through the layer
                input_data = layer2(input_data)
                if isinstance(layer2,nn.Linear) or isinstance(layer2,nn.Conv2d) or isinstance(layer2,nn.ConvTranspose2d):
                    previous_layer=layer2   
        else:
            if  isinstance(previous_layer,nn.Linear) and (isinstance(layer,nn.Conv2d) or isinstance(layer2,nn.ConvTranspose2d)):
                input_data=input_data.view(input_data.shape[0],layer.in_channels,int(np.sqrt(input_data.shape[1]/layer.in_channels)),-1)
            layer_flops = compute_flops(layer, input_data,i) 
            i=i+1
            forward_flops_dict[name] = layer_flops[0]
            backward_flops_dict[name] = layer_flops[1]
            # Pass the input through the layer
            input_data = layer(input_data)
            if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv2d) or isinstance(layer,nn.ConvTranspose2d):
                    previous_layer=layer   

    return forward_flops_dict,backward_flops_dict


def calculate_output_size(model, input_tensor):
    """Calculates the output size for each layer in the model and returns a dictionary with layer names as keys."""
    output_size_dict = {}
    input_data = input_tensor
    previous_layer = None
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            for name2, layer2 in layer.named_children():
                if isinstance(previous_layer, nn.Linear) and (isinstance(layer2, nn.Conv2d) or isinstance(layer2, nn.ConvTranspose2d)):
                    input_data = input_data.view(input_data.shape[0], layer2.in_channels, int(np.sqrt(input_data.shape[1] / layer2.in_channels)), -1)
                layer_output_size = compute_output_size(layer2, input_data)
                output_size_dict[name + "." + name2] = layer_output_size
                # Pass the input through the layer
                input_data = layer2(input_data)
                if isinstance(layer2, nn.Linear) or isinstance(layer2, nn.Conv2d) or isinstance(layer2, nn.ConvTranspose2d):
                    previous_layer = layer2   
        else:
            if isinstance(previous_layer, nn.Linear) and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)):
                input_data = input_data.view(input_data.shape[0], layer.in_channels, int(np.sqrt(input_data.shape[1] / layer.in_channels)), -1)
            layer_output_size = compute_output_size(layer, input_data)
            output_size_dict[name] = layer_output_size
            # Pass the input through the layer
            input_data = layer(input_data) 
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                previous_layer = layer

    return output_size_dict

#it compute the output size for each layer in bytes
def compute_output_size(layer, input_tensor):
    product=1
    for i in layer(input_tensor).shape:
        product=product*i
    return int(product * (32/8)) #multiply the number of output by 32 (float requires 32 bits) and divide by 8 to get the nummber of bytes


# this function sums every 3 elements in the list, it is used to sum the flops of every main layer with the batch normalization and the activation layer it follows, for simplification
def replace_with_sum(lst):
        result = []
        i = 0
        while i < len(lst):
            # Take 3 elements or the remaining elements if less than 3
            chunk = lst[i:i+3]
            result.append(sum(chunk)/1e9)
            i += 3
        return result

# This function keeps only in the lest the output of every third layer
def keep_every_third_element(lst):
    # Start from index 2 and select every third element
    result = lst[2::3]
    
    # If the number of elements left after the third element is less than 3, keep the last one
    if len(lst) % 3 != 0 and len(lst) > 2:
        result.append(lst[-1])
    
    return result
    
def get_flops_and_outputs():
    z_dim=100
    gener=Generator()
    disc=Discriminator()
    disc_input_tensor = torch.randn(64, 2, 28, 28)  
    gen_input_tensor=torch.randn(64,z_dim+10)
    forward_flops_disc,backward_flops_disc = calculate_flops(disc, disc_input_tensor)
    forward_flops_gen,backward_flops_gen = calculate_flops(gener, gen_input_tensor)


    gen_sizes=calculate_output_size(gener, gen_input_tensor)

    disc_sizes=calculate_output_size(disc, disc_input_tensor)
    
    #Here we sum the flops of every three layers
    #In doing so, we are making blocks of every three layers, which are usually a main layer like fc or conv or transposed conv followed by normalization and activation
    # the reason we do so is for simplification since normalization and activation contributes minimally to the flops, so we combine them with their main layer
    forward_gen_flops = replace_with_sum(list(forward_flops_gen.values()))


    backward_gen_flops = replace_with_sum(list(backward_flops_gen.values()))

    forward_disc_flops = replace_with_sum(list(forward_flops_disc.values()))

    backward_disc_flops = replace_with_sum(list(backward_flops_disc.values()))


    #we do the same here, as the cuts will be taken between the blocks
    gen_transmit_bytes=keep_every_third_element(list(gen_sizes.values()))

    disc_transmit_bytes=keep_every_third_element(list(disc_sizes.values()))
    
    #this is specific to our architecture, the output is in the form of [x1,x2,x3,x4]
    #x1 and x2 are the cutpoints for the generator head and tail, x3 and x4 are for the discriminator
    # 0 means no extra layer is added except for the necessary first block (in case of head), and last block (in case of tail), 1 mean, another extra block is added
    # they are only 0s and 1s, because both the generator and discriminator here, are composed of 5 blocks,the first always reside in the head, the last always in the tail, and the middle is always in the server segment
    # thus only 2 remaining blocks to locae per generator and discriminator, this should change if you made the architecture bigger
    possible_cuts = list(itertools.product([0, 1], repeat=4))
    flops_and_sizes_per_cuts={}

    for cut in possible_cuts:
        forward_flops_per_cut_client=(sum(forward_gen_flops[:cut[0]+1]),sum(forward_gen_flops[len(forward_gen_flops)-1-1*cut[1]:]),sum(forward_disc_flops[:cut[2]+1]),sum(forward_disc_flops[len(forward_disc_flops)-1-1*cut[3]:]))
        backward_flops_per_cut_client=(sum(backward_gen_flops[:cut[0]+1]),sum(backward_gen_flops[len(backward_gen_flops)-1-1*cut[1]:]),sum(backward_disc_flops[:cut[2]+1]),sum(backward_disc_flops[len(backward_disc_flops)-1-1*cut[3]:]))
        
        forward_flops_per_cut_server=(sum(forward_gen_flops[cut[0]+1:len(forward_gen_flops)-1-1*cut[1]]),sum(forward_disc_flops[cut[2]+1:len(forward_disc_flops)-1-1*cut[3]]))
        backward_flops_per_cut_server=(sum(backward_gen_flops[cut[0]+1:len(backward_gen_flops)-1-1*cut[1]]),sum(backward_disc_flops[cut[2]+1:len(backward_disc_flops)-1-1*cut[3]]))
        flops_per_cut_client=(forward_flops_per_cut_client,backward_flops_per_cut_client)
        flops_per_cut_server=(forward_flops_per_cut_server,backward_flops_per_cut_server)
        
        flops_and_sizes_per_cuts[cut]=(flops_per_cut_client,flops_per_cut_server,gen_transmit_bytes[cut[0]],gen_transmit_bytes[len(gen_transmit_bytes)-cut[1]-2],disc_transmit_bytes[cut[2]],disc_transmit_bytes[len(disc_transmit_bytes)-cut[3]-2])

    return flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops