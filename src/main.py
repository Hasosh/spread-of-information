import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a graph from edge list
G = nx.read_edgelist('politician_edges.csv', delimiter=',')


def spread_of_inormation(G, T, initial_adopt, type):
    G = G
    n = len(G.nodes())
    initial_adopt_num = int(initial_adopt * n)
    print(initial_adopt_num)

    # sort the nodes by degree
    node_and_degree = G.degree()
    (node, degree) = zip(*node_and_degree)
    sorted_deg = [x for (y,x) in sorted(zip(degree,node), reverse=True)]

    # get the page rank of the nodes with dumping factor 0.85
    pr = nx.pagerank(G, alpha=0.85)

    # sort the nodes by page rank
    sorted_pr = sorted(pr, key=pr.get, reverse=True)

    print(so)


    # add status & threshold attribute to nodes
    for node in G.nodes():
        G.nodes[node]['status'] = 0
        G.nodes[node]['threshold'] = random.random()

    # set initial adopters
    if type == 'central':
        for node in sorted_deg[:initial_adopt_num]:
            G.nodes[node]['status'] = 1
    elif type == 'marginal':
        for node in sorted_deg[-(initial_adopt_num)]:
            G.nodes[node]['status'] = 1
    else:
        for i in range(initial_adopt_num):
            node = random.choice(list(G.nodes()))
            G.nodes[node]['status'] = 1

    # spread of information

    T = T
    # if node does not have status 1 then calculate the perentage of it's neighbors that have status 1 and save that in the node attributes
    number_of_adopters = list()
    for i in range(T):
        for node in G.nodes():
            if G.nodes[node]['status'] == 0:
                neighbors = list(G.neighbors(node))
                count = 0
                for neighbor in neighbors:
                    if G.nodes[neighbor]['status'] == 1:
                        count += 1
                percentage = count/len(neighbors)
                G.nodes[node]['percentage_adopt'] = percentage

                # calculate the probability of adoption by taking the mean of perenctage and threshold, and save it as attribute
                if G.nodes[node]['percentage_adopt'] == 0:
                    G.nodes[node]['probability'] = 0
                else:
                    G.nodes[node]['probability'] = (G.nodes[node]['percentage_adopt'] + G.nodes[node]['threshold'])/2

                # flip a coin to update the status of the node
                if random.random() < G.nodes[node]['probability']:
                    G.nodes[node]['status'] = 1

        # get the list of nodes with status 1
        adopters = [node for node in G.nodes() if G.nodes[node]['status'] == 1]
        number_of_adopters.append(len(adopters))

    return number_of_adopters


ran = spread_of_inormation(G, 30, 0.025, 'random')
cen = spread_of_inormation(G, 30, 0.025, 'central')
mar = spread_of_inormation(G, 30, 0.025, 'marginal')

# plot the number of adopters over time
plt.plot(ran, label='random', color='red')
plt.plot(cen, label='central', color='blue')
plt.plot(mar, label='marginal', color='green')
plt.xlabel('Time')
plt.ylabel('Number of adopters')
plt.show()

