import torch
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
import copy
import torch.nn as nn

#this function represents the training iteration
def train_step(labels,clients_Gen1, clients_Gen2, server_gen,
               clients_Disc1, clients_Disc2, server_disc,
               clients_Gen1_optimizer, clients_Gen2_optimizer, server_gen_optimizer,
               clients_Disc2_optimizer, clients_Disc1_optimizer, server_disc_optimizer,
               real_images, device,client_cuts, z_dim=100,cluster=False,n_clusters=2):
    criterion = nn.BCELoss()
    labels=[labels[i].to(device) for i in range(len(labels))]
    batch_sizes = [ri.shape[0] for ri in real_images]
    real_images = [ri.to(device) for ri in real_images]
    real_labels = [torch.ones(batch_sizes[i], 1).to(device) for i in range(len(batch_sizes))]
    fake_labels = [torch.zeros(batch_sizes[i], 1).to(device)for i in range(len(batch_sizes))]
    # print(real_images)
    # --- Train Discriminator ---
    # Zero gradients for discriminator parts.
    server_disc_optimizer.zero_grad()
    for opt in clients_Disc1_optimizer:
        opt.zero_grad()
    for opt in clients_Disc2_optimizer:
        opt.zero_grad()

    fake_images=generate_fake(batch_sizes,z_dim,labels,clients_Gen1,server_gen,clients_Gen2,device,client_cuts)

    d_losses_fake,_,_,_=discriminate(batch_sizes,labels,clients_Disc1,server_disc,clients_Disc2,device,fake_labels,[fake_image.detach()for fake_image in fake_images],criterion,client_cuts)
    total_d_loss_fake=torch.stack(d_losses_fake).sum()
    total_d_loss_fake.backward()

    d_losses_real,cluster_labels,kld_scores_dict,global_kld_scores_dict=discriminate(batch_sizes,labels,clients_Disc1,server_disc,clients_Disc2,device,real_labels,real_images,criterion,client_cuts,cluster=cluster,n_clusters=n_clusters)
    total_d_loss_real=torch.stack(d_losses_real).sum()
    total_d_loss_real.backward()

    server_disc_optimizer.step()
    for opt in clients_Disc1_optimizer:
        opt.step()
    for opt in clients_Disc2_optimizer:
        opt.step()

    server_gen_optimizer.zero_grad()
    for opt in clients_Gen1_optimizer:
        opt.zero_grad()
    for opt in clients_Gen2_optimizer:
        opt.zero_grad()

    g_losses,_,_,_=discriminate(batch_sizes,labels,clients_Disc1,server_disc,clients_Disc2,device,real_labels,fake_images,criterion,client_cuts)
    total_g_loss=torch.stack(g_losses).sum()
    total_g_loss.backward()

    server_gen_optimizer.step()
    for opt in clients_Gen1_optimizer:
        opt.step()
    for opt in clients_Gen2_optimizer:
        opt.step()
    return torch.stack(d_losses_real).mean(), torch.stack(d_losses_fake).mean(),torch.stack(g_losses).mean(),cluster_labels,kld_scores_dict,global_kld_scores_dict



def generate_fake(batch_sizes,z_dim,image_labels,clients_Gen1,server_gen,clients_Gen2,device,client_cuts):
    client_gen1_outputs_0=[]
    client_gen1_outputs_1=[]
    g1_participants = [i for i in range(len(clients_Gen1)) if client_cuts[i][0] == 0]
    g2_participants = [i for i in range(len(clients_Gen1)) if client_cuts[i][0] != 0]
    # For a later stage, assume client_cuts[i][1]==0 means client i should leave.
    g3_participants = [i for i in range(len(clients_Gen1)) if client_cuts[i][1] == 0]
    for idx,client in enumerate(clients_Gen1):

        z = torch.randn(batch_sizes[idx], z_dim).to(device)
        if client_cuts[idx][0]==0:
            client_gen1_outputs_0.append(client(z,image_labels[idx]))
        else:
            client_gen1_outputs_1.append(client(z,image_labels[idx]))
    if client_gen1_outputs_0 !=[]:
        client_gen1_outputs_0=torch.cat(client_gen1_outputs_0,dim=0)
    else:
        client_gen1_outputs_0=torch.tensor([]).to(device)

    if client_gen1_outputs_1 !=[]:
        client_gen1_outputs_1=torch.cat(client_gen1_outputs_1,dim=0)
    else:
        client_gen1_outputs_1=torch.tensor([]).to(device)
    server_gen_outputs=server_gen(client_gen1_outputs_0,client_gen1_outputs_1,g1_participants,g2_participants,g3_participants,batch_sizes)
    clients_Gen2_outputs=[None]*len(clients_Gen2)

    for i in range(len(clients_Gen2)):

        clients_Gen2_outputs[i]=clients_Gen2[i](server_gen_outputs[i])


    return clients_Gen2_outputs

def discriminate(batch_sizes,image_labels,clients_Disc1,server_disc,clients_Disc2,device,labels,images,criterion,client_cuts,cluster=False,n_clusters=2):
    cluster_labels=None
    kld_scores_dict=None
    global_kld_scores_dict=None
    client_disc1_outputs_0=[]
    client_disc1_outputs_1=[]
    d1_participants = [i for i in range(len(clients_Disc1)) if client_cuts[i][2] == 0]
    d2_participants = [i for i in range(len(clients_Disc1)) if client_cuts[i][2] != 0]

    d3_participants = [i for i in range(len(clients_Disc1)) if client_cuts[i][3] == 0]

    for idx,client in enumerate(clients_Disc1):

        disc1_input=images[idx]
        if client_cuts[idx][2]==0:
            client_disc1_outputs_0.append(client(disc1_input,image_labels[idx]))
        else:
            client_disc1_outputs_1.append(client(disc1_input,image_labels[idx]))


    if client_disc1_outputs_0 !=[]:
        client_disc1_outputs_0=torch.cat(client_disc1_outputs_0,dim=0)
    else:
        client_disc1_outputs_0=torch.tensor([]).to(device)

    if client_disc1_outputs_1 !=[]:
        client_disc1_outputs_1=torch.cat(client_disc1_outputs_1,dim=0)
    else:
        client_disc1_outputs_1=torch.tensor([]).to(device)
    if not cluster:
        server_disc_outputs=server_disc(client_disc1_outputs_0,client_disc1_outputs_1,d1_participants,d2_participants,d3_participants,batch_sizes)
    else:
        server_disc_outputs,cluster_indeces,cluster_input=server_disc(client_disc1_outputs_0,client_disc1_outputs_1,d1_participants,d2_participants,d3_participants,batch_sizes,cluster=cluster)
        cluster_labels,kld_scores_dict,global_kld_scores_dict=cluster_tensors(cluster_input,n_clusters,cluster_indeces)
    clients_disc2_outputs=[None]*len(clients_Disc2)
    for i in range(len(clients_Disc2)):
        clients_disc2_outputs[i]=criterion(clients_Disc2[i](server_disc_outputs[i]),labels[i])
    return clients_disc2_outputs,cluster_labels,kld_scores_dict,global_kld_scores_dict



def cluster_tensors(inputs, num_clusters,clients_indeces):
    pooled_tensors = []
    pooled_tensors_t=[]
    for tensor in inputs:
        gap = torch.flatten(tensor.detach(), start_dim=1)
        gap_mean=torch.mean(gap,dim=0)# Global Average Pooling

        kld_tensor=torch.flatten(tensor.detach(), start_dim=2)
        kld_tensor=torch.mean(kld_tensor,dim=0)
        pooled_tensors.append(gap_mean.cpu().numpy())
        pooled_tensors_t.append(kld_tensor)

    all_features = np.array(pooled_tensors)



    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels_pred = kmeans.fit_predict(all_features)

     # Check output
    cluster_labels={}
    for i in range(len(cluster_labels_pred)):
        cluster_labels[clients_indeces[i]]=cluster_labels_pred[i]

    # print(f"Cluster labels: {cluster_labels}")
    kld_scores_dict={}
    for value in set(cluster_labels.values()):
        tensor_list=[]
        kld_indeces=[]
        for idx , i in enumerate(clients_indeces):
            if cluster_labels[i]==value:
                tensor_list.append(pooled_tensors_t[idx])
                kld_indeces.append(i)
        # print(f"tensor List_{tensor_list}")
        kld_scores=compute_kld_scores(tensor_list)

        for i in range(len(kld_scores)):
            kld_scores_dict[kld_indeces[i]]=kld_scores[i]

    global_kld_scores_dict={}
    global_kld_scores=compute_kld_scores(pooled_tensors_t)
    for i in range(len(global_kld_scores)):
            global_kld_scores_dict[clients_indeces[i]]=global_kld_scores[i]
    return cluster_labels,kld_scores_dict,global_kld_scores_dict




def compute_kld_scores(tensor_list):
    """
    Computes Kullback-Leibler Divergence (KLD) scores for each user (batch)
    against the global distribution (mean activation of all other users).

    :param tensor_list: List of PyTorch tensors, each of shape (Batch_size, N_filters, H, W)
    :return: List of KLD scores for each user
    """

    # Stack tensors to shape (N_users, Batch_size, N_filters, H, W)
    tensor_stack = torch.stack(tensor_list).float()  # Shape: (N_users, N_filters, H, W)

    # Convert activations into probability distributions using softmax
    user_probs = F.softmax(tensor_stack, dim=2)  # Ensure valid probability distributions

    kld_scores = []
    epsilon = 1e-9  # Small value to avoid log(0)

    for i in range(user_probs.shape[0]):
        P = user_probs[i]  # Current user’s probability distribution
        global_distribution = (user_probs.sum(dim=0) - P) / (user_probs.shape[0] - 1)  # Mean of others

        # Add epsilon to avoid log(0)
        P_safe = P + epsilon
        global_safe = global_distribution + epsilon

        # Compute KLD (P || Global)
        kld = F.kl_div(P_safe.log(), global_safe, reduction='batchmean')  # Sum over all elements
        kld_scores.append(kld.item())

    return kld_scores



def federated_averaging(client_GENs1,client_GENs2,client_Disc1,client_Disc2,server_GEN,server_Disc, client_sizes,all_layers_gen,all_layers_disc,smallest_components_clients,labels={},clients_scores={},global_scores={},alpha=1):
    printted_scores=[None]*len(client_GENs1)
    global_total_score=0
    all_layers_gen2=copy.deepcopy(all_layers_gen)
    all_layers_disc2=copy.deepcopy(all_layers_disc)
    if len(global_scores)>0:
        for i in range(len(client_sizes)):
            global_total_score=global_total_score+client_sizes[i]*np.exp(-1*alpha*global_scores[i])
    total_score=[0]*len(labels.values())

    for i in range(len(all_layers_disc2)):
        layer_sum=0

        for j in range(len(client_sizes)):

            if len(global_scores)==0 :
                score=client_sizes[j]/sum(client_sizes)
            else:
                # score=clients_scores[j]
                score=client_sizes[j]*np.exp(-1*alpha*global_scores[j])/global_total_score

            if len(client_Disc1[j].state_dict()) >i:
                layer_sum += score * client_Disc1[j].state_dict()[list(client_Disc1[j].state_dict().keys())[i]]
            elif len(all_layers_disc2) - len(client_Disc2[j].state_dict())  > i :
                layer_sum += score * server_Disc.state_dict()[list(all_layers_disc2.keys())[i]]
            else:
                layer_sum += score * client_Disc2[j].state_dict()[list(client_Disc2[j].state_dict().keys())[i-(len(all_layers_disc2) - len(client_Disc2[j].state_dict()))]]

        all_layers_disc2[list(all_layers_disc2.keys())[i]]=layer_sum

    for i in range(len(all_layers_gen2)):
        layer_sum=0

        for j in range(len(client_sizes)):

            if len(global_scores)==0 :
                score=client_sizes[j]/sum(client_sizes)
            else:
                # score=clients_scores[j]
                score=client_sizes[j]*np.exp(-1*alpha*global_scores[j])/global_total_score


            if len(client_GENs1[j].state_dict()) >i:
                layer_sum += score * client_GENs1[j].state_dict()[list(client_GENs1[j].state_dict().keys())[i]]
            elif len(all_layers_gen2) - len(client_GENs2[j].state_dict()) > i :
                layer_sum += score * server_GEN.state_dict()[list(all_layers_gen2.keys())[i]]
            else:
                # print("me: "+list(client_GENs2[j].state_dict().keys())[i-(len(all_layers_gen2) - len(client_GENs2[j].state_dict()))])
                layer_sum += score * client_GENs2[j].state_dict()[list(client_GENs2[j].state_dict().keys())[i-(len(all_layers_gen2) - len(client_GENs2[j].state_dict()))]]
        all_layers_gen2[list(all_layers_gen2.keys())[i]]=layer_sum

    for label in set(labels.values()):
        sizes=0
        for idx, val in labels.items():
            if val == label:
                sizes = sizes + client_sizes[idx]
                if len(clients_scores) >0:
                    total_score[label]=total_score[label]+client_sizes[idx]*np.exp(-1*alpha*clients_scores[idx])   # Accumulate sum
        # print(sizes)

        for i in range(len(all_layers_gen)):
            layer_sum=0

            # print("all: "+list(all_layers_gen.keys())[i])
            for j in range(len(client_sizes)):
                if labels[j]==label:
                    if len(clients_scores)==0 or sizes==client_sizes[j]:
                        if len(global_scores)==0 or sizes==client_sizes[j]:
                            score=client_sizes[j]/sizes
                        else:
                            score=client_sizes[j]*np.exp(-1*alpha*global_scores[j])/global_total_score
                        # print(f"client_{j}_size_{client_sizes[j]}_the score is {score}")
                    else:
                        score=(client_sizes[j]*np.exp(-1*alpha*clients_scores[j]) )/total_score[label]
                    # len_till_needed_server= len(Generator_extra_layers) -1*client_GENs2[j].additional_layers_count -server_gen_indeces[0] + len(client_GENs1[j].state_dict())
                    #The layer is in the first part of this client
                    printted_scores[j]=score
                    if len(client_GENs1[j].state_dict()) >i:
                        layer_sum += score * client_GENs1[j].state_dict()[list(client_GENs1[j].state_dict().keys())[i]]
                    elif len(all_layers_gen) - len(client_GENs2[j].state_dict()) > i :
                        layer_sum += score * server_GEN.state_dict()[list(all_layers_gen.keys())[i]]
                    else:
                        # print("me: "+list(client_GENs2[j].state_dict().keys())[i-(len(all_layers_gen) - len(client_GENs2[j].state_dict()))])
                        layer_sum += score * client_GENs2[j].state_dict()[list(client_GENs2[j].state_dict().keys())[i-(len(all_layers_gen) - len(client_GENs2[j].state_dict()))]]
            all_layers_gen[list(all_layers_gen.keys())[i]]=layer_sum

        for i in range(len(all_layers_disc)):
            layer_sum=0

            for j in range(len(client_sizes)):
                if labels[j]==label:
                    if len(clients_scores)==0 or sizes==client_sizes[j]:
                        if len(global_scores)==0 or sizes==client_sizes[j]:
                            score=client_sizes[j]/sizes
                        else:
                            score=client_sizes[j]*np.exp(-1*alpha*global_scores[j])/global_total_score
                    else:
                        score=(client_sizes[j]*np.exp(-1*alpha*clients_scores[j]) )/total_score[label]
                    # len_till_needed_server= len(Discriminator_extra_layers) -1*client_Disc2[j].additional_layers_count -server_disc_indeces[0] + len(client_Disc1[j].state_dict())
                    #The layer is in the first part of this client
                    if len(client_Disc1[j].state_dict()) >i:
                        layer_sum += score * client_Disc1[j].state_dict()[list(client_Disc1[j].state_dict().keys())[i]]
                    elif len(all_layers_disc) - len(client_Disc2[j].state_dict())  > i :
                        layer_sum += score * server_Disc.state_dict()[list(all_layers_disc.keys())[i]]
                    else:
                        layer_sum += score * client_Disc2[j].state_dict()[list(client_Disc2[j].state_dict().keys())[i-(len(all_layers_disc) - len(client_Disc2[j].state_dict()))]]

            all_layers_disc[list(all_layers_disc.keys())[i]]=layer_sum

        for i in range(len(client_sizes)):
                if labels[i]==label:

                    zip1=zip(client_GENs1[i].state_dict().keys(),list(all_layers_gen.values())[:len(client_GENs1[i].state_dict())])
                    zip1=dict(zip1)
                    client_GENs1[i].load_state_dict(zip1)
                    zip1=zip(client_GENs2[i].state_dict().keys(),list(all_layers_gen.values())[-1*len(client_GENs2[i].state_dict()):])
                    zip1=dict(zip1)
                    client_GENs2[i].load_state_dict(zip1)

                    zip1=zip(client_Disc1[i].state_dict().keys(),list(all_layers_disc.values())[:len(client_Disc1[i].state_dict())])
                    zip1=dict(zip1)
                    client_Disc1[i].load_state_dict(zip1)
                    zip1=zip(client_Disc2[i].state_dict().keys(),list(all_layers_disc.values())[-1*len(client_Disc2[i].state_dict()):])
                    zip1=dict(zip1)
                    client_Disc2[i].load_state_dict(zip1)

    server_GEN.load_state_dict(dict(list(all_layers_gen2.items())[len(client_GENs1[smallest_components_clients[0]].state_dict()):-1*len(client_GENs2[smallest_components_clients[1]].state_dict())-1]))
    server_Disc.load_state_dict(dict(list(all_layers_disc2.items())[len(client_Disc1[smallest_components_clients[2]].state_dict()):-1*len(client_Disc2[smallest_components_clients[3]].state_dict())-1]))
