from Data import data
from models import get_models , Generator_extra_layers
from HuSCFGAN import train_step,federated_averaging
from Metrics.evaluation import get_dataset_classifiers,calculate_image_score,generate_classifier_data,get_accuracy
import yaml
import torch 
import matplotlib.pyplot as plt
import torchvision
import pandas as pd

import os
import pandas as pd
import matplotlib.pyplot as plt

def save_results(scenario, mnist_scores=None, fmnist_scores=None, kmnist_scores=None, notmnist_scores=None,
                 latency=None, metrics_dict=None):
    
    scenario_dir = f"./Results/Scenario_{scenario}"
    plots_dir = os.path.join(scenario_dir, "plots")
    tables_dir = os.path.join(scenario_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    def plot_scores(scores, title, filename):
        if scores is not None:
            plt.figure()
            plt.plot(range(len(scores)), scores, marker='o')
            plt.title(title)
            plt.xlabel("Round")
            plt.ylabel("Score")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, filename))
            plt.close()

    # Plot and save per-round scores
    if mnist_scores != []:
        plot_scores(mnist_scores, "MNIST Scores per Round", "mnist_scores.png")
    if fmnist_scores != []:
        plot_scores(fmnist_scores, "Fashion MNIST Scores per Round", "fmnist_scores.png")
    if kmnist_scores != []:
        plot_scores(kmnist_scores, "KMNIST Scores per Round", "kmnist_scores.png")
    if notmnist_scores != []:
        plot_scores(notmnist_scores, "NotMNIST Scores per Round", "notmnist_scores.png")

    # Save final metrics as CSV
    if metrics_dict:
        pd.DataFrame(metrics_dict).to_csv(os.path.join(tables_dir, "final_metrics.csv"), index=False)

    # Save latency
    if latency is not None:
        pd.DataFrame({"latency (s)": [latency]}).to_csv(os.path.join(tables_dir, "latency.csv"), index=False)

with open("./configs.yaml", "r") as f:
    config = yaml.safe_load(f)

    # Extract values from the nested structure
clients = config["clients"]
training = config["training"]
experiment = config["experiment"]
settings = config["settings"]
# Assign to variables
num_digit_clients = clients["num_digit_clients"]
num_fashion_clients = clients["num_fashion_clients"]
num_kmnist_clients = clients["num_kmnist_clients"]
num_notmnist_clients = clients["num_notmnist_clients"]

batch_size = training["batch_size"]
scenario = experiment["scenario"]
num_clients = num_digit_clients + num_fashion_clients +num_kmnist_clients + num_notmnist_clients 
num_epochs =settings['local_epochs']
num_rounds=settings['num_rounds']

n_clusters=int(num_digit_clients>0)+int(num_fashion_clients>0)+int(num_kmnist_clients>0)+int(num_notmnist_clients>0)

mnist_scores=[]
fmnist_scores=[]
kmnist_scores=[]
notmnist_scores=[]

clients_data_loaders,client_sizes,test_loaders,digit_exclusions,fashion_exclusions,kmnist_exclusions,letter_exclusions=data.get_data_loaders()
num_batches = max(len(dl) for dl in clients_data_loaders)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

client_GENs1, client_GENs2, server_GEN, client_Disc1, client_Disc2, server_Disc, \
client_gen1_optimizers, client_gen2_optimizers, server_GEN_optimizer, \
client_disc1_optimizers, client_disc2_optimizers, server_Disc_optimizer,all_layers_gen, \
all_layers_disc, smallest_components_clients, client_cuts,server_gen_indeces,latency = get_models(device)

for round in range(num_rounds):
    for epoch in range(num_epochs):

        for batch_idx in range(num_batches):
            
            real_images=[]
            image_labels=[]
            for client_idx, data_loader in enumerate(clients_data_loaders):

                if epoch ==num_epochs-1 and batch_idx==num_batches-1:
                    cluster=True
                else:
                    cluster=False

                # Process one batch per client
                batch_data, labels = next(iter(data_loader))
                real_images.append(batch_data)
                image_labels.append(labels)


            d_loss_real,d_loss_fake, g_loss,labels,kld_scores_dict,global_kld_scores_dict = train_step(image_labels,
                client_GENs1, client_GENs2, server_GEN,
                client_Disc1, client_Disc2, server_Disc,
                client_gen1_optimizers, client_gen2_optimizers,
                server_GEN_optimizer, client_disc2_optimizers,
                client_disc1_optimizers, server_Disc_optimizer,
                real_images, device,client_cuts,cluster=cluster,n_clusters=n_clusters
            )

        print(f"Epoch {epoch + 1}/{num_epochs}, Round {round+1}/{num_rounds}, "
        f"Average D Loss Real: {d_loss_real.item():.4f}, "
        f"Average D Loss Fake: {d_loss_fake.item():.4f}, "
        f"Average G Loss: {g_loss.item():.4f}, ")

    with torch.no_grad():
        for m in range(num_clients):
            if m%4==0:
                if m<num_digit_clients:
                    excluded=digit_exclusions.get(m,[])
                elif m<num_fashion_clients+num_digit_clients:
                    excluded=fashion_exclusions.get(m,[])
                elif m<num_notmnist_clients+num_fashion_clients+num_digit_clients:
                    excluded=letter_exclusions.get(m,[])

                im_labels = torch.empty(64, dtype=torch.long).to(device)

                for i in range(64):
                     # Get excluded labels for client `i`, default to an empty set
                    while True:
                        label = torch.randint(0, 10, (1,)).item()  # Generate a random label
                        if label not in excluded:
                            im_labels[i] = label
                            break

                c1=client_GENs1[m](torch.randn(64, 100).to(device),im_labels)
                c2=server_GEN.forward_c(c1,[client_GENs1[m].additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_GENs2[m].additional_layers_count -server_gen_indeces[0] -1 ])

                fake_images = client_GENs2[m](c2)
                fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]
                grid = torchvision.utils.make_grid(fake_images)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                # plt.savefig(f"local_generated_images_client_{m+1}_round_{round+1}_epoch_{epoch+1}.png")
                plt.close()

    


    if round <=1 :
        labels={key:0 for key in range(num_clients)}
        kld_scores_dict={}
        global_kld_scores_dict={}


    mnist_classifier,fmnist_classifier,kmnist_classifier,notmnist_classifier=get_dataset_classifiers(device)
    federated_averaging(client_GENs1,client_GENs2,client_Disc1,client_Disc2,server_GEN,server_Disc,client_sizes,all_layers_gen,all_layers_disc,smallest_components_clients,labels=labels,clients_scores=kld_scores_dict,global_scores=global_kld_scores_dict,alpha=1)
    if num_digit_clients>0:
        mnist_scores.append(calculate_image_score(mnist_classifier,client_GENs1[0],client_GENs2[0],server_GEN,device,server_gen_indeces,Generator_extra_layers,type='mnist'))
    if num_fashion_clients>0:
      fmnist_scores.append(calculate_image_score(fmnist_classifier,client_GENs1[num_digit_clients],client_GENs2[num_digit_clients],server_GEN,device,server_gen_indeces,Generator_extra_layers,type='fmnist'))
    if num_kmnist_clients>0:
      kmnist_scores.append(calculate_image_score(kmnist_classifier,client_GENs1[num_digit_clients+num_fashion_clients],client_GENs2[num_digit_clients+num_fashion_clients],server_GEN,device,server_gen_indeces,Generator_extra_layers,type='kmnist'))
    if num_notmnist_clients>0:
      notmnist_scores.append(calculate_image_score(notmnist_classifier,client_GENs1[num_digit_clients+num_fashion_clients+num_kmnist_clients],client_GENs2[num_digit_clients+num_fashion_clients+num_kmnist_clients],server_GEN,device,server_gen_indeces,Generator_extra_layers,type='notmnist'))

    if num_digit_clients>0:
        with torch.no_grad():
            im_labels = torch.randint(0, 10, (64,)).to(device)
            c1=client_GENs1[0](torch.randn(64, 100).to(device),im_labels)
            c2=server_GEN.forward_c(c1,[client_GENs1[0].additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_GENs2[0].additional_layers_count -server_gen_indeces[0] -1 ])

            fake_images = client_GENs2[0](c2)
            fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]

            grid = torchvision.utils.make_grid(fake_images)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig(f"./Visualization/Federated_generated_images_round_{round+1}_cluster_{int(num_digit_clients>0)}.png")
            plt.close()
    if num_fashion_clients>0:
        with torch.no_grad():
            im_labels = torch.randint(0, 10, (64,)).to(device)
            c1=client_GENs1[num_digit_clients](torch.randn(64, 100).to(device),im_labels)
            c2=server_GEN.forward_c(c1,[client_GENs1[num_digit_clients].additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_GENs2[num_digit_clients].additional_layers_count -server_gen_indeces[0] -1 ])

            fake_images = client_GENs2[num_digit_clients](c2)
            fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]
            grid = torchvision.utils.make_grid(fake_images)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig(f"./Visualization/Federated_generated_images_round_{round+1}_cluster_{int(num_digit_clients>0)+int(num_fashion_clients>0)}.png")
            plt.close()

    if num_kmnist_clients>0:
        with torch.no_grad():
            im_labels = torch.randint(0, 10, (64,)).to(device)
            c1=client_GENs1[num_digit_clients+num_fashion_clients](torch.randn(64, 100).to(device),im_labels)
            c2=server_GEN.forward_c(c1,[client_GENs1[num_digit_clients+num_fashion_clients].additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_GENs2[num_digit_clients+num_fashion_clients].additional_layers_count -server_gen_indeces[0] -1 ])

            fake_images = client_GENs2[num_digit_clients+num_fashion_clients](c2)
            fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]
            grid = torchvision.utils.make_grid(fake_images)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig(f"./Visualization/Federated_generated_images_round_{round+1}_cluster_{int(num_digit_clients>0)+int(num_fashion_clients>0)+int(num_kmnist_clients>0)}.png")
            plt.close()

    if num_notmnist_clients>0:
        with torch.no_grad():
            im_labels = torch.randint(0, 10, (64,)).to(device)
            c1=client_GENs1[num_digit_clients+num_fashion_clients+num_kmnist_clients](torch.randn(64, 100).to(device),im_labels)
            c2=server_GEN.forward_c(c1,[client_GENs1[num_digit_clients+num_fashion_clients+num_kmnist_clients].additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_GENs2[num_digit_clients+num_fashion_clients+num_kmnist_clients].additional_layers_count -server_gen_indeces[0] -1 ])

            fake_images = client_GENs2[num_digit_clients+num_fashion_clients+num_kmnist_clients](c2)
            fake_images = fake_images * 0.5 + 0.5  # Denormalize to [0, 1]
            grid = torchvision.utils.make_grid(fake_images)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig(f"./Visualization/Federated_generated_images_round_{round+1}_cluster_{int(num_digit_clients>0)+int(num_fashion_clients>0)+int(num_kmnist_clients>0)+int(num_notmnist_clients>0)}.png")
            plt.close()


metrics = []

if num_digit_clients > 0:
    generate_classifier_data(client_GENs1[0],client_GENs2[0],server_GEN,device,server_gen_indeces,Generator_extra_layers,num_samples=30000)
    mnist_metrics = get_accuracy(test_loaders['mnist'])
    metrics.append({"Dataset": "MNIST", **dict(zip(["Accuracy", "Precision", "Recall", "F1", "FPR"], mnist_metrics))})

if num_fashion_clients > 0:
    generate_classifier_data(client_GENs1[num_digit_clients],client_GENs2[num_digit_clients],server_GEN,device,server_gen_indeces,Generator_extra_layers,num_samples=30000,type='fmnist')
    fmnist_metrics = get_accuracy(test_loaders['fmnist'], type='fmnist')
    metrics.append({"Dataset": "FashionMNIST", **dict(zip(["Accuracy", "Precision", "Recall", "F1", "FPR"], fmnist_metrics))})

if num_kmnist_clients > 0:
    generate_classifier_data(client_GENs1[num_digit_clients+num_fashion_clients],client_GENs2[num_digit_clients+num_fashion_clients],server_GEN,device,server_gen_indeces,Generator_extra_layers,num_samples=30000,type='kmnist')
    kmnist_metrics = get_accuracy(test_loaders['kmnist'], type='kmnist')
    metrics.append({"Dataset": "KMNIST", **dict(zip(["Accuracy", "Precision", "Recall", "F1", "FPR"], kmnist_metrics))})

if num_notmnist_clients > 0:
    generate_classifier_data(client_GENs1[num_digit_clients+num_fashion_clients+num_kmnist_clients],client_GENs2[num_digit_clients+num_fashion_clients+num_kmnist_clients],server_GEN,device,server_gen_indeces,Generator_extra_layers,num_samples=30000,type='notmnist')
    notmnist_metrics = get_accuracy(test_loaders['notmnist'], type='notmnist')
    metrics.append({"Dataset": "NotMNIST", **dict(zip(["Accuracy", "Precision", "Recall", "F1", "FPR"], notmnist_metrics))})


# Save all results
save_results(
    scenario=scenario,
    mnist_scores=mnist_scores,
    fmnist_scores=fmnist_scores,
    kmnist_scores=kmnist_scores,
    notmnist_scores=notmnist_scores,
    latency=latency,
    metrics_dict=metrics
)
    


