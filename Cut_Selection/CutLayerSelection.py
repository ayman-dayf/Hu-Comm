import yaml
import random
import numpy as np
from collections import Counter
with open("./configs.yaml", "r") as f:
    config = yaml.safe_load(f)

clients = config["clients"]
# Extract the number of clients
num_digit_clients = clients["num_digit_clients"]
num_fashion_clients = clients["num_fashion_clients"]
num_kmnist_clients = clients["num_kmnist_clients"]
num_notmnist_clients = clients["num_notmnist_clients"]
num_clients = num_digit_clients + num_fashion_clients +num_kmnist_clients + num_notmnist_clients 


#the fitness function calculate the iteration latency over all devices in the system according to the architecture in the given paper
def fitness(individual,flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops,client_specs,computational_frequency_server,server_cpu_cycles_per_flop,server_transmission_rate,alpha=0.5,gamma=0.5):
    computational_frequency_hz_server = computational_frequency_server
    
    #the variables and array lengths here are specific to our architecture, not dynamic
    
    max_genhead_forward_latency_c=[0]*2 #[x1,x2] where x_i is the maximum latency for the clients who has their cut point in the head part at this position, if the architecture is bigger, the array is bigger
    forward_gen_latency_s=[0]*3  #[s1,s2,s3] where s1 is the latency for server computation of the second block for all participating clients in it and s2 for third block and s3 for forth, it should be bigger if the architecture is bigger
    max_gentail_forward_latency_c=0 #the max time needed for the tail computation 

    max_gentail_backward_latency_c=[0]*2 #[x1,x2] where x_i is the maximum latency for the clients who has their cut point in the tail part at this position, if the architecture is bigger, the array is bigger
    backward_gen_latency_s=[0]*3  #[s1,s2,s3] where s1 is the latency for server computation of the second block for all participating clients in it and s2 for third block and s3 for forth, it should be bigger if the architecture is bigger
    max_genhead_backward_latency_c=0 #the max time needed for the head computation 
    
    #Number of clients participating in the 2nd block layer at the server side, (They dont have it locally, where head cutpoint is before it)
    count_2nd_layer_gen_participants = sum(1 for x in individual if x[0] == 0)
    #Number of clients participating in the 4th block layer at the server side, (They dont have it locally, where tail cutpoint is after it)
    count_4th_layer_gen_participants = sum(1 for x in individual if x[1] == 0)
    
    max_dischead_forward_latency_c=[0]*2 #[x1,x2] where x_i is the maximum latency for the clients who has their cut point in the head part at this position, if the architecture is bigger, the array is bigger
    forward_disc_latency_s=[0]*3  #[s1,s2,s3] where s1 is the latency for server computation of the second block for all participating clients in it and s2 for third block and s3 for forth, it should be bigger if the architecture is bigger
    max_disctail_forward_latency_c=0 #the max time needed for the tail computation either locally or at server
    
    max_disctail_backward_latency_c=[0]*2 #[x1,x2] where x_i is the maximum latency for the clients who has their cut point in the tail part at this position, if the architecture is bigger, the array is bigger
    backward_disc_latency_s=[0]*3  #[s1,s2,s3] where s1 is the latency for server computation of the second block for all participating clients in it and s2 for third block and s3 for forth, it should be bigger if the architecture is bigger
    max_dischead_backward_latency_c=0 #the max time needed for the head computation 
    
    #Number of clients participating in the 2nd block layer at the server side, (They dont have it locally, where head cutpoint is before it)
    count_2nd_layer_disc_participants = sum(1 for x in individual if x[2] == 0)
    #Number of clients participating in the 4th block layer at the server side, (They dont have it locally, where tail cutpoint is after it)
    count_4th_layer_disc_participants = sum(1 for x in individual if x[3] == 0)
    
    #generator heads latencies per client
    client_gen1_forward_computational_latencies=[0]*num_clients
    client_gen1_backward_computational_latencies=[0]*num_clients
    client_gen1_forward_transmission_latencies=[0]*num_clients

    #generator tails latencies per client
    client_gen2_forward_computational_latencies=[0]*num_clients
    client_gen2_backward_computational_latencies=[0]*num_clients
    client_gen2_backward_transmission_latencies=[0]*num_clients

    #discriminator heads latencies per client
    client_disc1_forward_computational_latencies=[0]*num_clients
    client_disc1_backward_computational_latencies=[0]*num_clients
    client_disc1_forward_transmission_latencies=[0]*num_clients

    #discriminator tails latencies per client
    client_disc2_forward_computational_latencies=[0]*num_clients
    client_disc2_backward_computational_latencies=[0]*num_clients
    client_disc2_backward_transmission_latencies=[0]*num_clients


    server_gen_forward_transmission_latency=0
    server_disc_forward_transmission_latency=0
    server_gen_backward_transmission_latency=0
    server_disc_backward_transmission_latency=0
    
    forward_gen_latency_s[1]=forward_gen_flops[2]*len(individual) / (computational_frequency_hz_server*server_cpu_cycles_per_flop)
    forward_gen_latency_s[2]=forward_gen_flops[3]*count_4th_layer_gen_participants / (computational_frequency_hz_server*server_cpu_cycles_per_flop)
    
    forward_disc_latency_s[1]=forward_disc_flops[2]*len(individual)/(computational_frequency_hz_server*server_cpu_cycles_per_flop)
    forward_disc_latency_s[2]=forward_disc_flops[3]*count_4th_layer_disc_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)
    
    backward_gen_latency_s[1]=backward_gen_flops[2]*len(individual)/(computational_frequency_hz_server*server_cpu_cycles_per_flop)
    backward_gen_latency_s[0]=backward_gen_flops[3]*count_2nd_layer_gen_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)
    
    backward_disc_latency_s[1]=backward_disc_flops[2]*len(individual)/(computational_frequency_hz_server*server_cpu_cycles_per_flop)
    backward_disc_latency_s[0]=backward_disc_flops[3]*count_2nd_layer_disc_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)
    for i,client in enumerate(individual):
        parameters=flops_and_sizes_per_cuts[client]
        computational_frequency_hz_client = client_specs[i][0] * 1e-3
        
        
        #Total Latency
        # Calculate the forward pass latency for client
        client_gen1_forward_computational_latencies[i] = parameters[0][0][0]  / (computational_frequency_hz_client*client_specs[i][1]) 
        client_gen2_forward_computational_latencies[i] = parameters[0][0][1]  / (computational_frequency_hz_client*client_specs[i][1]) 
        client_disc1_forward_computational_latencies[i] = 3*parameters[0][0][2]  / (computational_frequency_hz_client*client_specs[i][1]) 
        client_disc2_forward_computational_latencies[i] = 3*parameters[0][0][3]  / (computational_frequency_hz_client*client_specs[i][1]) 
        #Calculate Client Transmission Latencies
        client_gen1_forward_transmission_latencies[i] = parameters[2] /client_specs[i][2]
        client_disc1_forward_transmission_latencies[i]=3*parameters[4] /client_specs[i][2]
        
        #Calculate Server transmission Latencies
        server_gen_forward_transmission_latency = parameters[3] / server_transmission_rate
        server_disc_forward_transmission_latency = 3*parameters[5] / server_transmission_rate
        
        if (max_genhead_forward_latency_c[client[0]] <client_gen1_forward_computational_latencies[i]+ client_gen1_forward_transmission_latencies[i]):
            max_genhead_forward_latency_c[client[0]]=client_gen1_forward_computational_latencies[i]+ client_gen1_forward_transmission_latencies[i]
        
        if client[1]==1:
            max_gentail_forward_latency_c=max(max_gentail_forward_latency_c, server_gen_forward_transmission_latency + client_gen2_forward_computational_latencies[i])
        else:
            max_gentail_forward_latency_c=max(max_gentail_forward_latency_c, forward_gen_latency_s[2]+server_gen_forward_transmission_latency + client_gen2_forward_computational_latencies[i])
        
        if (max_dischead_forward_latency_c[client[2]] <client_disc1_forward_computational_latencies[i]+ client_disc1_forward_transmission_latencies[i]):
            max_dischead_forward_latency_c[client[2]]=client_disc1_forward_computational_latencies[i]+ client_disc1_forward_transmission_latencies[i]
        
        if client[3]==1:
            max_disctail_forward_latency_c=max(max_disctail_forward_latency_c, server_disc_forward_transmission_latency + client_disc2_forward_computational_latencies[i])
        else:
            max_disctail_forward_latency_c=max(max_disctail_forward_latency_c, forward_disc_latency_s[2]+server_disc_forward_transmission_latency + client_disc2_forward_computational_latencies[i])
        
         
        #Calculate Backward pass Latency for the client
        client_gen1_backward_computational_latencies[i] = parameters[0][1][0]  / (computational_frequency_hz_client*client_specs[i][1]) 
        client_gen2_backward_computational_latencies[i] = parameters[0][1][1]  / (computational_frequency_hz_client*client_specs[i][1]) 
        client_disc1_backward_computational_latencies[i] = 3*parameters[0][1][2]  / (computational_frequency_hz_client*client_specs[i][1]) 
        client_disc2_backward_computational_latencies[i] = 3*parameters[0][1][3]  / (computational_frequency_hz_client*client_specs[i][1]) 
        #Calculate Client Transmission Latencies
        client_gen2_backward_transmission_latencies[i] = parameters[3] /client_specs[i][2]
        client_disc2_backward_transmission_latencies[i]=3*parameters[5] /client_specs[i][2]
 
        server_gen_backward_transmission_latency = parameters[2] / server_transmission_rate
        server_disc_backward_transmission_latency = 3*parameters[4] / server_transmission_rate
        
        
        if (max_gentail_backward_latency_c[client[1]] <client_gen2_backward_computational_latencies[i]+ client_gen2_backward_transmission_latencies[i]):
            max_gentail_backward_latency_c[client[1]]=client_gen2_backward_computational_latencies[i]+ client_gen2_backward_transmission_latencies[i]
        
        if client[0]==1:
            max_genhead_backward_latency_c=max(max_genhead_backward_latency_c, server_gen_backward_transmission_latency + client_gen1_backward_computational_latencies[i])
        else:
            max_genhead_backward_latency_c=max(max_genhead_backward_latency_c, backward_gen_latency_s[0]+server_gen_backward_transmission_latency + client_gen1_backward_computational_latencies[i])
        
        if (max_disctail_backward_latency_c[client[3]] <client_disc2_backward_computational_latencies[i]+ client_disc2_backward_transmission_latencies[i]):
            max_disctail_backward_latency_c[client[3]]=client_disc2_backward_computational_latencies[i]+ client_disc2_backward_transmission_latencies[i]
        
        if client[2]==1:
            max_dischead_backward_latency_c=max(max_dischead_backward_latency_c, server_disc_backward_transmission_latency + client_disc1_backward_computational_latencies[i])
        else:
            max_dischead_backward_latency_c=max(max_dischead_backward_latency_c, backward_disc_latency_s[0]+server_disc_backward_transmission_latency + client_disc1_backward_computational_latencies[i])
         
    #the total time before the next layer, is the max computation of this layer of server and the other clients computing locally
    forward_gen_latency_s[0]=max(forward_gen_flops[1]*count_2nd_layer_gen_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop) + max_genhead_forward_latency_c[0], max_genhead_forward_latency_c[1] ) 
    
    forward_disc_latency_s[0]=max(forward_disc_flops[1]*count_2nd_layer_disc_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop) + max_dischead_forward_latency_c[0], max_dischead_forward_latency_c[1] ) 
    
    backward_gen_latency_s[2]=max(backward_gen_flops[3]*count_4th_layer_gen_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop) + max_gentail_backward_latency_c[0], max_gentail_backward_latency_c[1] ) 
    
    backward_disc_latency_s[2]=max(backward_disc_flops[3]*count_4th_layer_disc_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop) + max_disctail_backward_latency_c[0], max_disctail_backward_latency_c[1] ) 
    
    
    Total_forward_gen_time= forward_gen_latency_s[0]+forward_gen_latency_s[1]+max_gentail_forward_latency_c
    Total_forward_disc_time= forward_disc_latency_s[0]+forward_disc_latency_s[1]+max_disctail_forward_latency_c
    
    Total_backward_gen_time= backward_gen_latency_s[2]+backward_gen_latency_s[1]+max_genhead_backward_latency_c
    Total_backward_disc_time= backward_disc_latency_s[2]+backward_disc_latency_s[1]+max_dischead_backward_latency_c
    
    Total_forward_time=Total_forward_gen_time+Total_forward_disc_time
    
    Total_backward_time=Total_backward_gen_time+Total_backward_disc_time
    
    Total_training_time=Total_forward_time+Total_backward_time
    
    server_utiliztion_time=((forward_gen_flops[1]*count_2nd_layer_gen_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)) + forward_gen_latency_s[1]+forward_gen_latency_s[2]+
                            (forward_disc_flops[1]*count_2nd_layer_disc_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)) + forward_disc_latency_s[1]+forward_disc_latency_s[2]+
                            (backward_gen_flops[3]*count_4th_layer_gen_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)) + backward_gen_latency_s[1]+backward_gen_latency_s[0]+
                            (backward_disc_flops[3]*count_4th_layer_disc_participants/(computational_frequency_hz_server*server_cpu_cycles_per_flop)) + backward_disc_latency_s[1]+backward_disc_latency_s[0])
    

    return alpha*(Total_training_time-server_utiliztion_time)  + gamma*server_utiliztion_time

def get_best_cuts(flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops):
    # Load YAML
    with open("./Cut_Selection/profiles.yaml", "r") as f:
        data = yaml.safe_load(f)

    # Extract server config
    server_config = data.pop("server")  # remove server block from device list

    computational_frequency_server = server_config["computational_frequency_ghz"]
    server_cpu_cycles_per_flop = server_config["cpu_cycles_per_flop"]
    server_transmission_rate = server_config["transmission_rate_bps"]


    # Build the profiles list
    profiles = []

    for _, specs in data.items():
        cpu_mhz = specs["cpu_mhz"]
        flops = specs["flops_per_cycle"]
        data_rate = specs["data_rate"]
        profiles.append((cpu_mhz, flops,data_rate))



    num_devices = len(profiles)
    clients_per_device = num_clients // num_devices
    remainder = num_clients % num_devices

    # Start with equal distribution
    client_specs = [profile for profile in profiles for _ in range(clients_per_device)]

    # Add remainder randomly
    extra_profiles = random.sample(profiles, remainder)
    client_specs.extend(extra_profiles)

    # Shuffle the final list
    random.shuffle(client_specs)


    # scale=4
    # num_clients_red=num_clients//scale
    # Problem parameters
    POPULATION_SIZE = 1000
    # NUM_TUPLES = num_clients       # Number of tuples per individual (can be changed)
    # NUM_TUPLES = num_clients_red
    NUM_TUPLES = len(profiles) 
    TUPLE_LENGTH = 4     # Length of each tuple
    GENERATIONS = 1000
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.01

    

    # def downsample_preserving_ratio(arr, new_length=num_clients_red):
        
    #     n = len(arr)
    #     if new_length >= n:
    #         return arr.copy()
        
    #     # Count frequencies of each unique value.
    #     freq = Counter(arr)
        
    #     # Compute the number of elements for each unique value based on the ratio.
    #     # We'll use rounding to get integer counts.
    #     new_counts = {val: round((count / n) * new_length) for val, count in freq.items()}
        
    #     # Adjust total count if rounding didn't sum exactly to new_length.
    #     total = sum(new_counts.values())
    #     # If we're short, add one to random keys until total equals new_length.
    #     while total < new_length:
    #         key = random.choice(list(new_counts.keys()))
    #         new_counts[key] += 1
    #         total += 1
    #     # If we have too many, remove one from random keys with count > 0.
    #     while total > new_length:
    #         key = random.choice(list(new_counts.keys()))
    #         if new_counts[key] > 0:
    #             new_counts[key] -= 1
    #             total -= 1

    #     # For each unique value, randomly sample the assigned number of items from the original array.
    #     result = []
    #     for val, count in new_counts.items():
    #         # Gather all occurrences of val.
    #         val_items = [item for item in arr if item == val]
    #         # Sample 'count' items without replacement.
    #         if count > 0:
    #             sampled = random.sample(val_items, count)
    #             result.extend(sampled)
        
    #     # Optionally, shuffle to mix the values.
    #     random.shuffle(result)
    #     return result


    def flatten_individual(individual):
        """
        Converts a list of tuples (each containing bits) into a single string of bits.
        Example:
        [(0,1,0,1), (1,0,1,0)]  -> "01011010"
        """
        return ''.join(''.join(str(bit) for bit in tup) for tup in individual)

    def unflatten_individual(bit_string, num_tuples, tuple_length):
        """
        Converts a bit string back into a list of tuples.
        Example:
        "01011010" with num_tuples=2 and tuple_length=4 -> [(0,1,0,1), (1,0,1,0)]
        """
        return [tuple(int(bit) for bit in bit_string[i * tuple_length:(i + 1) * tuple_length])
                for i in range(num_tuples)]


    # def expand_indvidual(individual):
    #     expanded_ind=individual*scale
    #     return expanded_ind

    def expand_indvidual(individual,specs=client_specs,profiles=profiles):
        expanded_ind=[0]*len(specs)
        for i,spec in enumerate(specs):
            for j,profile in enumerate(profiles):
                if spec==profile:
                    expanded_ind[i]=individual[j]
                    break
        return expanded_ind


    def generate_individual(num_tuples=NUM_TUPLES, tuple_length=TUPLE_LENGTH):
        """
        Creates a random individual represented as a list of tuples.
        Each tuple is a bit-string (tuple of 0s and 1s).
        """
        return [tuple(random.randint(0, 1) for _ in range(tuple_length)) for _ in range(num_tuples)]



    def tournament_selection(population,flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops,client_specs,computational_frequency_server,server_cpu_cycles_per_flop,server_transmission_rate,alpha=1,gamma=0.5, k=5):
        """
        Selects one individual from the population using tournament selection.
        """
        
        tournament = random.sample(population, k)
        tournament.sort(key=lambda individual: fitness(expand_indvidual(individual),flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops,client_specs,computational_frequency_server,server_cpu_cycles_per_flop,server_transmission_rate,alpha=alpha,gamma=gamma))
        return tournament[0]


    def crossover(parent1, parent2):
        """
        Converts the parent's list of tuples into a single bit string,
        performs uniform crossover, then converts the resulting strings
        back into the original list-of-tuples format.
        
        Assumes both parents have the same structure (same number of tuples
        and same tuple lengths).
        """
        # Flatten the parents to bit strings.
        flat1 = flatten_individual(parent1)
        flat2 = flatten_individual(parent2)
        if random.random()<CROSSOVER_RATE:
            if random.random() < 0.5:
            # Perform uniform crossover on the flat representation.
            
                child1_str = "".join(flat1[i] if random.random() < 0.5 else flat2[i] for i in range(len(flat1)))
                child2_str = "".join(flat2[i] if random.random() < 0.5 else flat1[i] for i in range(len(flat2)))
                
                    
            else:
                point1 = random.randint(1, len(flat1) - 2)
                point2 = random.randint(point1 + 1, len(flat1) - 1)
                child1_str = flat1[:point1] + flat2[point1:point2] + flat1[point2:]
                child2_str = flat2[:point1] + flat1[point1:point2] + flat2[point2:]
        else:
            child1_str, child2_str = flat1, flat2
    

        # Recover the original structure.
        num_tuples = len(parent1)
        tuple_length = len(parent1[0])
        child1 = unflatten_individual(child1_str, num_tuples, tuple_length)
        child2 = unflatten_individual(child2_str, num_tuples, tuple_length)
        
        return child1, child2

    def mutate(individual):
        """
        Mutates an individual by flipping bits in each tuple with a small probability.
        Since tuples are immutable, we convert them to lists, perform mutation, then convert back.
        """
        mutated_individual = []
        for t in individual:
            mutated_tuple = []
            for bit in t:
                if random.random() < MUTATION_RATE:
                    mutated_tuple.append(1 - bit)
                else:
                    mutated_tuple.append(bit)
            mutated_individual.append(tuple(mutated_tuple))
        return mutated_individual

    def genetic_algorithm(client_specs=client_specs,alpha=1,gamma=1):
        
        # Initialize population
        population = [generate_individual() for _ in range(POPULATION_SIZE)]
        best=None

        # client_specs=downsample_preserving_ratio(client_specs)*scale
        for generation in range(GENERATIONS):

            new_population = []
            
            # Create new population using selection, crossover, and mutation
            while len(new_population) < POPULATION_SIZE:
                parent1 = tournament_selection(population,flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops,client_specs,computational_frequency_server,server_cpu_cycles_per_flop,server_transmission_rate,alpha=alpha,gamma=gamma)
                parent2 = tournament_selection(population,flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops,client_specs,computational_frequency_server,server_cpu_cycles_per_flop,server_transmission_rate,alpha=alpha,gamma=gamma)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(mutate(child2))
            
            population = new_population
            
            # Find the best individual in the current generation
            min=np.inf
            
            best=None
            for individual in population:
                x=fitness(expand_indvidual(individual),flops_and_sizes_per_cuts,forward_gen_flops,forward_disc_flops,backward_gen_flops,backward_disc_flops,client_specs,computational_frequency_server,server_cpu_cycles_per_flop,server_transmission_rate,alpha=alpha,gamma=gamma)
                if x < min:
                    min=x

                    best=individual
            print(f"Generation {generation+1}/{GENERATIONS} , Latency={min} s")
    
        return expand_indvidual(best),min
    return genetic_algorithm(alpha=1,gamma=1)

