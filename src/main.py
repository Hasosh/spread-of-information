import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter, NullFormatter # for log ticks
from scipy.optimize import curve_fit # for bass model

def show_prop(G):
    # normal graph properties
    print(f'Number of nodes: {G.number_of_nodes()}')
    print(f'Number of edges: {G.number_of_edges()}')
    if nx.is_connected(G):
        print(f'Clustering coefficient: {nx.transitivity(G)}')
        print(f'Diameter: {nx.diameter(G)}')
        print(f'Average path length: {nx.average_shortest_path_length(G)}')
        print(f'Graph density: {nx.density(G)}')

    # draw degree rank plot
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    
    fig, ax = plt.subplots()

    ax.loglog(degrees, degree_freq,'go-')
    ax.set_title("Degree Distribution (Log-Log Scale)")
    ax.set_ylabel('Degree')
    ax.set_xlabel('Frequency')
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    fig.tight_layout()
    plt.show()

def generate_random_graph(G=None, nodes=None, edges=None, random_type="ER"):
    # compute number of nodes and edges
    if G is not None:
        n = G.number_of_nodes()
        m = G.number_of_edges()
    else:
        n = nodes
        m = edges

    # generate desired random graph
    if random_type == "ER":
        # Erdös Renyi random graph
        g_ran = nx.gnm_random_graph(n=n, m=m, directed=False)
    elif random_type == "WS":
        # Watts Strogatz random graph: rewiring with p = 0.01: high clustering and low diameter
        g_ran = nx.watts_strogatz_graph(n=n, k=int(m/n), p=0.01) 
    elif random_type == "BA":
        # Barabasi Albert random graph
        g_ran = nx.barabasi_albert_graph(n=n, m=int(m/n))
    elif random_type == "KB":
        # Kleinberg model random graph
        lattice_length = math.ceil(math.sqrt(n))
        g_ran = nx.navigable_small_world_graph(n=lattice_length, p=1, q=1, r=10, dim=2)
    elif random_type == "newman_WS":
        # Newmann-Watts-Strogatz random graph
        g_ran = nx.newman_watts_strogatz_graph(n=n, k=int((m) / n * 2), p=0.01)
    else:
        print("please specify one of ER / WS / BA / KB")
        return
    g_ran = g_ran.to_undirected()
    return(g_ran)

def graph_preprocessing(G):
    """for a given input graph, the graph is changed such that it is undirected, has no self-loops, and no disconnected components"""

    # make graph undirected
    G = G.to_undirected()
    
    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # remove disconnected components
    if not nx.is_connected(G):
        components = [a for a in nx.connected_components(G)]
        if len(components) > 1:
            biggest_component = len(components[0])
            for c in components[1:]:
                if len(c) > biggest_component:
                    biggest_component = len(c)
            for c in components:
                if len(c) != biggest_component:
                    for node in c:
                        G.remove_node(node)
        assert nx.is_connected(G)
    return G

def fit_bass_model(data):
    """fits the bass model on given xdata and ydata"""
    def c_t(x, p, q, m):
        return (p+(q/m)*(x))*(m-x)
    popt, pcov = curve_fit(c_t, xdata=range(len(data)), ydata=data)
    p, q, m = popt[0], popt[1], popt[2]
    return c_t(range(len(data)), p, q, m)

def spread_of_information(graph, timestamp, percentage_initial_adopt, type_initial_adopt, centrality_measure):
    """
      Simulates the spread of information on a graph based on the given parameters.

      Parameters:
      - graph (networkx Graph): the graph on which the simulation will be run
      - timestamp (int): number of rounds the simulation will run for
      - percentage_initial_adopt (float): the percentage of nodes that will start as adopters, between 0 and 1
      - type_initial_adopt (str): the method of selecting initial adopters, either "central", "marginal" or "random"
      - centrality_measure (str): the method of sorting the nodes, either "degree", "pagerank" or "closeness"

      Returns:
      - number_of_adopters (list): list containing the number of adopters after each round of the simulation

      """
    G = graph
    n = len(G.nodes())
    initial_adopt_num = int(percentage_initial_adopt * n)

    if centrality_measure == 'degree':
        # sort the nodes by degree
        node_and_degree = G.degree()
        (node, degree) = zip(*node_and_degree)
        sorted_nodes = [x for (y,x) in sorted(zip(degree,node), reverse=True)]
    elif centrality_measure == 'pagerank':
        # sort the nodes by pagerank
        pr = nx.pagerank(G, alpha=0.85)
        sorted_nodes = sorted(pr, key=pr.get, reverse=True)
    elif centrality_measure == 'closeness':
        # sort the nodes by closeness
        cc = nx.closeness_centrality(G)
        sorted_nodes = sorted(cc, key=cc.get, reverse=True)

    # eigenvector centrality for each node
    ec = nx.eigenvector_centrality(G)
    # normalize the eigenvector centrality to be between 0 and 1
    ec = {k: v / max(ec.values()) for k, v in ec.items()}

    # add status & threshold attribute to nodes
    for node in G.nodes():
        G.nodes[node]['status'] = 0
        G.nodes[node]['threshold'] = random.random()
        G.nodes[node]['influence'] = ec[node]


    # set initial adopters
    if type_initial_adopt == 'central':
        for node in sorted_nodes[:initial_adopt_num]:
            G.nodes[node]['status'] = 1
    elif type_initial_adopt == 'marginal':
        for node in sorted_nodes[-(initial_adopt_num)]:
            G.nodes[node]['status'] = 1
    else:
        for i in range(initial_adopt_num):
            node = random.choice(list(G.nodes()))
            G.nodes[node]['status'] = 1

    # spread of information
    t = timestamp
    # list of adopters
    number_of_adopters = [initial_adopt_num]
    for i in range(t):
        # if full diffusion happened then stop the process
        if i != 0 and number_of_adopters[-1] == n:
            # fill the list with the last value which is the number of nodes
            number_of_adopters += [n] * (t - len(number_of_adopters) + 1) 
            #print(f'Diffusion already stopped at time step: {i+1}')
            # break out of the outer for loop
            break
        # iterate over nodes
        for node in G.nodes():
            # if the node is not an adopter
            if G.nodes[node]['status'] == 0:
                # get neighbors and check if they are adopters, calculate the percentage of neighbors that are adopters
                neighbors = list(G.neighbors(node))
                count = 0
                total_influence = 0
                for neighbor in neighbors:
                    if G.nodes[neighbor]['status'] == 1:
                        count += 1
                        # total influence of adopters, normalized by the number of neighbors
                        total_influence += G.nodes[neighbor]['influence']
                percentage = count/len(neighbors)
                G.nodes[node]['percentage_adopt'] = percentage
                G.nodes[node]['neighbor_influence'] = total_influence / len(neighbors)

                # calculate the probability of adoption (mean of percentage and threshold)
                if G.nodes[node]['percentage_adopt'] == 0:
                    G.nodes[node]['probability'] = 0
                else:
                    G.nodes[node]['probability'] = (G.nodes[node]['percentage_adopt'] +
                                                    G.nodes[node]['threshold'] + G.nodes[node]['neighbor_influence'])/3

                # sample from a uniform distribution and check if the probability is greater than the sample
                if random.random() < G.nodes[node]['probability']:
                    G.nodes[node]['status'] = 1

        # get list of adopters
        adopters = [node for node in G.nodes() if G.nodes[node]['status'] == 1]
        number_of_adopters.append(len(adopters))

    return number_of_adopters

def make_diffusions_and_average(simulations, graph, timestamp, percentage_initial_adopt, type_initial_adopt, centrality_measure):
    """simulates several diffusions for the SAME graph and averages the results"""
    cumulated_diffusion_list = list()
    for i in range(simulations):
        print(f"Simulation {i+1} of {simulations}")
        diff = spread_of_information(graph, timestamp, percentage_initial_adopt, type_initial_adopt, centrality_measure)
        if i==0:
            # initialize with values first
            cumulated_diffusion_list = diff
        else:
            # add new diff to cumulated diff
            cumulated_diffusion_list = [sum(x) for x in zip(cumulated_diffusion_list, diff)]
    averaged_diffusion_list = [x / simulations for x in cumulated_diffusion_list]
    print("All simulations completed!")
    return averaged_diffusion_list

# Create a graph from edge list
G = nx.read_edgelist('data/facebook_combined.txt', delimiter=' ') # ../data/facebook_combined.txt

# preprocess the graph
G = graph_preprocessing(G)

# show properties
show_prop(G)

# simulation parameters
simulations = 5
time_steps = 20
percentage_initial_adopt = 0.025

# run the simulation
ran = make_diffusions_and_average(simulations, G, time_steps, percentage_initial_adopt, 'random', 'degree') # 5 simulations averaged (for the SAME graph)
cen = make_diffusions_and_average(simulations, G, time_steps, percentage_initial_adopt, 'central', 'degree') # 5 simulations averaged (for the SAME graph)
mar = make_diffusions_and_average(simulations, G, time_steps, percentage_initial_adopt, 'marginal', 'degree') # 5 simulations averaged (for the SAME graph)

# ran = spread_of_information(G, 20, 0.025, 'random', 'degree')
# cen = spread_of_information(G, 20, 0.025, 'central', 'degree')
# mar = spread_of_information(G, 20, 0.025, 'marginal', 'degree')
print(len(ran), len(cen), len(mar))

# get percentage of adopters
ran_perc = [x/len(G.nodes()) for x in ran]
cen_perc = [x/len(G.nodes()) for x in cen]
mar_perc = [x/len(G.nodes()) for x in mar]

# plot the number of adopters over time with seaborn
df = pd.DataFrame({'random': ran_perc, 'central': cen_perc, 'marginal': mar_perc})
sns.lineplot(data=df, palette="tab10", linewidth=2.5, dashes=False)
plt.xlabel('Time')
plt.ylabel('Number of Adopters')

# fit of bass model
fig, ax = plt.subplots()
ax.plot(ran_perc, label="Random")
ax.plot(fit_bass_model(ran_perc), label="Bass model")
ax.set_xlabel('Time')
ax.set_ylabel('Fit of bass model')

plt.legend()
plt.show()



