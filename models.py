import torch
import torch.nn as nn
import numpy as np
from Cut_Selection import CutLayerSelection,modelCutter
import yaml
import copy
import torch.optim as optim


#the client generator head model, containing the necessary first block, composed of the embedding layer and the first fc layer, and the possibility to add layers (client specific)
class ClientGen(nn.Module):
    def __init__(self, z_dim=100,num_classes=10):
        super(ClientGen, self).__init__()
        # Initial layers
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(z_dim+num_classes, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.additional_layers_cg1 = nn.ModuleList()
        self.additional_layers_count=0

    def add_layer(self, layer):

        self.additional_layers_cg1.append(layer)
        if not hasattr(layer,'reset_parameters'):
            for layer_child in layer.children():
                if hasattr(layer_child, 'reset_parameters'):
                    layer_child.reset_parameters()
        else:
            layer.reset_parameters()
        self.additional_layers_count=self.additional_layers_count+1

    def forward(self, x,labels):
        # Pass through the initial layers
        label_embed=self.label_embedding(labels)
        x=torch.cat([x, label_embed], dim=1)
        x = self.fc(x).view(-1, 256, 7, 7)  # Reshape as needed
        # Pass through additional layers
        for layer in self.additional_layers_cg1:
            x = layer(x)
        return x

#server generator segment with the middle necessary layer, and the possibility to add sequel or prequal layers
class ServerGen(nn.Module):
    def __init__(self):
        super(ServerGen, self).__init__()
        self.additional_layers_sg = nn.ModuleList()
        self.additional_layers_count=0

    def add_layer(self, layer):
        self.additional_layers_sg.append(layer)
        if not hasattr(layer,'reset_parameters'):
            for layer_child in layer.children():
                if hasattr(layer_child, 'reset_parameters'):
                    layer_child.reset_parameters()
        else:
            layer.reset_parameters()
        self.additional_layers_count=self.additional_layers_count+1

    def forward(self, x0,x1,g1_participants,g2_participants,g3_participants,batch_sizes):
        server_outputs=[None]*(len(g1_participants+g2_participants))
        if len(x0)>0:
            x0=self.additional_layers_sg[0](x0)
            x=self.additional_layers_sg[1](torch.cat((x0,x1),dim=0))
        else:
            x=self.additional_layers_sg[0](x1)
        all_g_participants=g1_participants+g2_participants
        selected_sizes = np.array(batch_sizes)[np.array(all_g_participants)].tolist()
        chunks_g2 = torch.split(x, selected_sizes, dim=0)
        g3_input=[]
        g3_input_indeces=[]
        for i,client_i in enumerate(all_g_participants.copy()):
            if client_i not in g3_participants:
                server_outputs[client_i]=chunks_g2[i]
            else:
                g3_input.append(chunks_g2[i])
                g3_input_indeces.append(client_i)
        if len(g3_input)>0:
            g3_output=self.additional_layers_sg[-1](torch.cat(g3_input,dim=0))
            selected_sizes = np.array(batch_sizes)[np.array(g3_input_indeces)].tolist()
            chunks_g3 = torch.split(g3_output, selected_sizes, dim=0)
            for i,client_i in enumerate(g3_input_indeces):
                server_outputs[client_i]=chunks_g3[i]
        return server_outputs

    def forward_c(self, x,server_cut_indeces):
        i=0
        for layer in self.additional_layers_sg:
            if (i >= server_cut_indeces[0]) and (i <= server_cut_indeces[1]) :
                x = layer(x)
            i=i+1
        return x

# th client generator tail, with the necessary final block, with the possibility of adding previous layer according to the device the capabilities
class ClientGen2(nn.Module):
    def __init__(self):
        super(ClientGen2, self).__init__()
        self.additional_layers_cg2 = nn.ModuleList()
        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(64, 1, 3, 1, 1),  # (Batch, 1, 28, 28)
            nn.Tanh()
        )

        self.additional_layers_count=0

    def add_layer(self, layer):
        self.additional_layers_cg2.append(layer)
        if not hasattr(layer,'reset_parameters'):
            for layer_child in layer.children():
                if hasattr(layer_child, 'reset_parameters'):
                    layer_child.reset_parameters()
        else:
            layer.reset_parameters()
        self.additional_layers_count=self.additional_layers_count+1

    def forward(self, x):
        for layer in self.additional_layers_cg2:
            x = layer(x)
        x = self.deconv(x)  # Flatten the input
        return x

class ClientDisc(nn.Module):
    def __init__(self,z_dim=100,num_classes=10):
        super(ClientDisc, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 28*28)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            # (Batch, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.additional_layers_cd1 = nn.ModuleList()
        self.additional_layers_count=0

    def add_layer(self, layer):
        self.additional_layers_cd1.append(layer)
        if not hasattr(layer,'reset_parameters'):
            for layer_child in layer.children():
                if hasattr(layer_child, 'reset_parameters'):
                    layer_child.reset_parameters()
        else:
            layer.reset_parameters()
        self.additional_layers_count=self.additional_layers_count+1


    def forward(self, x,labels):
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(-1, 1, 28, 28)
        x = torch.cat([x, label_embed], dim=1)
        x = self.conv(x)
        for layer in self.additional_layers_cd1:
            x = layer(x)
        return x

class ServerDisc(nn.Module):
    def __init__(self):
        super(ServerDisc, self).__init__()
        self.additional_layers_sd = nn.ModuleList()
        self.additional_layers_count=0

    def add_layer(self, layer):
        self.additional_layers_sd.append(layer)
        if not hasattr(layer,'reset_parameters'):
            for layer_child in layer.children():
                if hasattr(layer_child, 'reset_parameters'):
                    layer_child.reset_parameters()
        else:
            layer.reset_parameters()
        self.additional_layers_count=self.additional_layers_count+1

    def forward(self, x0,x1,d1_participants,d2_participants,d3_participants,batch_sizes,cluster=False):
        server_outputs=[None]*(len(d1_participants+d2_participants))
        if len(x0)>0:
            x0=self.additional_layers_sd[0](x0)
            x=self.additional_layers_sd[1](torch.cat((x0,x1),dim=0))
        else:
            x=self.additional_layers_sd[0](x1)
        all_d_participants=d1_participants+d2_participants
        selected_sizes = np.array(batch_sizes)[np.array(all_d_participants)].tolist()
        chunks_d2 = torch.split(x, selected_sizes, dim=0)
        d3_input=[]
        d3_input_indeces=[]
        for i,client_i in enumerate(all_d_participants.copy()):
            if client_i not in d3_participants:
                server_outputs[client_i]=chunks_d2[i]
            else:
                d3_input.append(chunks_d2[i])
                d3_input_indeces.append(client_i)
        if len(d3_input)>0:
            d3_output=self.additional_layers_sd[-1](torch.cat(d3_input,dim=0))
            selected_sizes = np.array(batch_sizes)[np.array(d3_input_indeces)].tolist()
            chunks_d3 = torch.split(d3_output, selected_sizes, dim=0)
            for i,client_i in enumerate(d3_input_indeces):
                server_outputs[client_i]=chunks_d3[i]
        if cluster:
            return server_outputs, all_d_participants.copy(),[chunk.clone() for chunk in chunks_d2]
        else:
            return server_outputs

    def forward_c(self, x,server_cut_indeces):
        i=0
        for layer in self.additional_layers_sd:
            if (i >= server_cut_indeces[0]) and (i <= server_cut_indeces[1]) :
                x = layer(x)
            i=i+1
        return x

class ClientDisc2(nn.Module):
    def __init__(self):
        super(ClientDisc2, self).__init__()
        self.additional_layers_cd2 = nn.ModuleList()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(256 * 3 * 3, 1) ) # Latent space to output
        self.sigmoid=nn.Sigmoid()

        self.additional_layers_count=0

    def add_layer(self, layer):
        self.additional_layers_cd2.append(layer)
        if not hasattr(layer,'reset_parameters'):
            for layer_child in layer.children():
                if hasattr(layer_child, 'reset_parameters'):
                    layer_child.reset_parameters()
        else:
            layer.reset_parameters()
        self.additional_layers_count=self.additional_layers_count+1

    def forward(self, x):
        for layer in self.additional_layers_cd2:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)


# the generator discriminator extra layers, which are distributed according to device capabilities to the client and servere
Generator_extra_layers = [
    # First block: Upsample to 14x14x128
    nn.Sequential(
        nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Upsample to 14x14x128
        nn.BatchNorm2d(128),
        nn.ReLU(True)
    ),

    # Middle block: Same input and output shape as the first block
    nn.Sequential(
        nn.ConvTranspose2d(128, 128, 3, 1, 1),  # Kernel size 3, stride 1, padding 1 to preserve shape
        nn.BatchNorm2d(128),
        nn.ReLU(True)
    ),

    # Second block: Upsample to 28x28x64
    nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, 2, 1),  # Upsample to 28x28x64
        nn.BatchNorm2d(64),
        nn.ReLU(True)
    )  # Output between [-1, 1]
]

Discriminator_extra_layers = [
    # First block: Downsample to 7x7x128
    nn.Sequential(
        nn.Conv2d(64, 128, 4, 2, 1),  # (Batch, 128, 7, 7)
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True)
    ),

    # Middle block: Same input and output shape as the first block
    nn.Sequential(
        nn.Conv2d(128, 128, 3, 1, 1),  # Kernel size 3, stride 1, padding 1 to preserve shape
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True)
    ),

    # Second block: Downsample to 3x3x256
    nn.Sequential(
        nn.Conv2d(128, 256, 4, 2, 1),  # (Batch, 256, 3, 3)
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True)
    ),
]

def get_models(device):
    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)
        clients = config["clients"]
        num_digit_clients = clients["num_digit_clients"]
        num_fashion_clients = clients["num_fashion_clients"]
        num_kmnist_clients = clients["num_kmnist_clients"]
        num_notmnist_clients = clients["num_notmnist_clients"]
        num_clients = num_digit_clients + num_fashion_clients +num_kmnist_clients + num_notmnist_clients 
        lr = config["optimizer"]["learning_rate"]
        betas = tuple(config["optimizer"]["betas"]) 
    

    flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops=modelCutter.get_flops_and_outputs()
    best_cuts,latency=CutLayerSelection.get_best_cuts(flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops)
    # Initialize models

    client_GENs1=[]
    client_GENs2=[]
    client_Disc1=[]
    client_Disc2=[]



    client_GENs1 = [ClientGen().to(device) for _ in range(num_clients)]
    server_GEN= ServerGen().to(device)
    client_GENs2 = [ClientGen2().to(device) for _ in range(num_clients)]

    client_Disc1 = [ClientDisc().to(device) for _ in range(num_clients)]
    server_Disc= ServerDisc().to(device)
    client_Disc2 = [ClientDisc2().to(device) for _ in range(num_clients)]



    server_gen_indeces=[np.inf,-1*np.inf]
    server_disc_indeces=[np.inf,-1*np.inf]
    smallest_components_clients = [min(col) for col in zip(*best_cuts)]#an index for smallest client index for each part
    client_cuts=[]

    for i in range(num_clients):
        #for each client generate thhe cuts for generator and discriminator
        client_cuts.append(best_cuts[i%len(best_cuts)])
        front_layers_gen= best_cuts[i%len(best_cuts)][0]
        back_layers_gen=best_cuts[i%len(best_cuts)][1]
        front_layers_disc=best_cuts[i%len(best_cuts)][2]
        back_layers_disc=best_cuts[i%len(best_cuts)][3]

        for j in range(front_layers_gen):
            client_GENs1[i].add_layer(copy.deepcopy(Generator_extra_layers[j].to(device)))

        for j in range(back_layers_gen):
            client_GENs2[i].add_layer(copy.deepcopy(Generator_extra_layers[-1*(back_layers_gen-j)].to(device)))

        for j in range(front_layers_disc):
            client_Disc1[i].add_layer(copy.deepcopy(Discriminator_extra_layers[j].to(device)))

        for j in range(back_layers_disc):
            client_Disc2[i].add_layer(copy.deepcopy(Discriminator_extra_layers[-1*(back_layers_disc-j)].to(device)))

        #assign the smallest needed server part to the server (server indices are the part of the additional layers assigned to the server)
        if (front_layers_gen < server_gen_indeces[0]):
            server_gen_indeces[0]=front_layers_gen
            smallest_components_clients[0]=i
        if (len(Generator_extra_layers)-back_layers_gen-1 > server_gen_indeces[1]):
            server_gen_indeces[1]=len(Generator_extra_layers)-back_layers_gen-1
            smallest_components_clients[1]=i
        if (front_layers_disc < server_disc_indeces[0]):
            server_disc_indeces[0]=front_layers_disc
            smallest_components_clients[2]=i
        if (len(Discriminator_extra_layers)-back_layers_disc-1 > server_disc_indeces[1]):
            server_disc_indeces[1]=len(Discriminator_extra_layers)-back_layers_disc-1
            smallest_components_clients[3]=i

    for i in range(server_gen_indeces[0],server_gen_indeces[1]+1):
        server_GEN.add_layer(copy.deepcopy(Generator_extra_layers[i].to(device)))

    for i in range(server_disc_indeces[0],server_disc_indeces[1]+1):
        server_Disc.add_layer(copy.deepcopy(Discriminator_extra_layers[i].to(device)))

    for i in range(num_clients):
        client_GENs1[i]=client_GENs1[i].to(device)
        client_GENs2[i]=client_GENs2[i].to(device)
        client_Disc1[i]=client_Disc1[i].to(device)
        client_Disc2[i]=client_Disc2[i].to(device)

    all_layers_gen=list(client_GENs1[smallest_components_clients[0]].state_dict().keys())+list(server_GEN.state_dict().keys())+list(client_GENs2[smallest_components_clients[1]].state_dict().keys())
    all_layers_disc=list(client_Disc1[smallest_components_clients[2]].state_dict().keys())+list(server_Disc.state_dict().keys())+list(client_Disc2[smallest_components_clients[3]].state_dict().keys())
    all_layers_gen = {key: None for key in all_layers_gen}
    all_layers_disc = {key: None for key in all_layers_disc}

    client_gen1_optimizers = [optim.Adam(model.parameters(), lr=lr, betas=betas) for model in client_GENs1]
    client_gen2_optimizers = [optim.Adam(model.parameters(),lr=lr, betas=betas) for model in client_GENs2]
    client_disc1_optimizers = [optim.Adam(model.parameters(),lr=lr, betas=betas) for model in client_Disc1]
    client_disc2_optimizers = [optim.Adam(model.parameters(), lr=lr, betas=betas) for model in client_Disc2]

    server_GEN_optimizer = optim.Adam(server_GEN.parameters(), lr=lr, betas=betas)
    server_Disc_optimizer = optim.Adam(server_Disc.parameters(), lr=lr, betas=betas)

    return client_GENs1,client_GENs2,server_GEN,client_Disc1,client_Disc2,server_Disc,client_gen1_optimizers,client_gen2_optimizers,server_GEN_optimizer,client_disc1_optimizers,client_disc2_optimizers,server_Disc_optimizer,all_layers_gen,all_layers_disc,smallest_components_clients,client_cuts,server_gen_indeces,latency