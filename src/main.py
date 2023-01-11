import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd

# Create a graph from edge list
G = nx.read_edgelist('../data/facebook_combined.txt', delimiter=' ')

def spread_of_inormation(graph, timestamp, percentage_initial_adopt, type_initial_adopt, centrality_measure):
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

# run the simulation
ran = spread_of_inormation(G, 30, 0.025, 'random', 'degree')
cen = spread_of_inormation(G, 30, 0.025, 'central', 'degree')
mar = spread_of_inormation(G, 30, 0.025, 'marginal', 'degree')

# get percentage of adopters
ran_perc = [x/len(G.nodes()) for x in ran]
cen_perc = [x/len(G.nodes()) for x in cen]
mar_perc = [x/len(G.nodes()) for x in mar]

# plot the number of adopters over time with seaborn
df = pd.DataFrame({'random': ran_perc, 'central': cen_perc, 'marginal': mar_perc})
sns.lineplot(data=df, palette="tab10", linewidth=2.5, dashes=False)
plt.xlabel('Time')
plt.ylabel('Number of Adopters')
plt.show()




