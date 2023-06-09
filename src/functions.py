import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy.optimize import curve_fit


def show_and_save_prop(G, name):
    """
    This function takes a graph G as an input and prints & saves various properties of the graph
    such as the number of nodes, edges, clustering coefficient, diameter, average path length,
    graph density and plots the degree distribution of the graph on a log-log scale.

    Parameters
    ----------
    G : graph
        A NetworkX graph object
     name : string
         Name of the graph

    Returns
    -------
    None
    """
    # graph properties
    n = G.number_of_nodes()
    m = G.number_of_edges()
    # graph connectedness needs to be checked for the following measures, otherwise error is thrown
    if nx.is_connected(G):
        cc = nx.transitivity(G)
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        density = nx.density(G)
    else:
        cc, diameter, avg_path_length, density = -1, -1, -1, -1

    # create or append results to csv file
    if not os.path.exists(f"../results/{name}"):
        # create a new directory for the graph
        os.makedirs(f"../results/{name}")
        # create a dataframe with the properties and save it as csv
        df = pd.DataFrame(
            {
                "n": [n],
                "m": [m],
                "cc": [cc],
                "diameter": [diameter],
                "avg_path_length": [avg_path_length],
                "density": [density],
            }
        )
        df.to_csv(f"../results/{name}/{name}_properties.csv", index=False)
    else:
        # append the properties to the csv file
        df = pd.read_csv(f"../results/{name}/{name}_properties.csv")
        df = df.append(
            {
                "n": n,
                "m": m,
                "cc": cc,
                "diameter": diameter,
                "avg_path_length": avg_path_length,
                "density": density,
            },
            ignore_index=True,
        )
        df.to_csv(f"../results/{name}/{name}_properties.csv", index=False)

    print(f"Graph {name} has {n} nodes, {m} edges")
    if nx.is_connected(G):
        print(
            f"Graph {name} has clustering coefficient {cc}, diameter {diameter}, average path length {avg_path_length} and density {density}"
        )

    # draw degree rank plot
    if name not in ["ER", "WS", "BA", "newman_WS"]:
        degree_freq = nx.degree_histogram(G)
        degrees = range(len(degree_freq))

        fig, ax = plt.subplots()

        ax.loglog(degrees, degree_freq, "go-")
        ax.set_title(f"Degree Distribution of {name} dataset (Log-Log Scale)")
        ax.set_ylabel("Degree")
        ax.set_xlabel("Frequency")
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

        fig.tight_layout()
        plt.show(block=False)
        fig.savefig(f"../results/{name}/{name}_degree_distribution.png")


def generate_random_graph(G=None, random_type="ER", nodes=None, edges=None):
    """
    This function takes a graph G as an input and generates a random graph
    of the specified type and number of nodes and edges.
    Parameters
    ----------
    G : graph
        A NetworkX graph object
    nodes : int
        The number of nodes in the random graph
    edges : int
        The number of edges in the random graph
    random_type : str
        The type of random graph to generate. The options are "ER" (Erdös Renyi random graph),
        "WS" (Watts Strogatz random graph), "BA" (Barabasi Albert random graph),
        "KB" (Kleinberg model random graph) and "newman_WS" (Newmann-Watts-Strogatz random graph).
    Returns
    -------
    g_ran : graph
        A NetworkX graph object of the generated random graph
    """
    # compute number of nodes and edges
    if G is not None:
        n, m = G.number_of_nodes(), G.number_of_edges()
    else:
        n, m = nodes, edges
    assert (
        n is not None and m is not None
    ), "Please provide valid input for number of nodes and edges"

    # generate desired random graph
    random_type_dict = {
        "ER": nx.gnm_random_graph,
        "WS": nx.watts_strogatz_graph,
        "BA": nx.barabasi_albert_graph,
        "KB": nx.navigable_small_world_graph,
        "newman_WS": nx.newman_watts_strogatz_graph,
    }
    if random_type not in random_type_dict:
        raise ValueError(
            f"Invalid random_type: {random_type}. Please choose one of ER / WS / BA / KB / newman_WS"
        )
    if random_type == "ER":
        # Erdös Renyi random graph
        g_ran = random_type_dict[random_type](n=n, m=m)
    elif random_type == "WS":
        # Watts Strogatz random graph: rewiring with p = 0.01: high clustering and low diameter
        g_ran = random_type_dict[random_type](n=n, k=int(m / n) * 2, p=0.01)
    elif random_type == "BA":
        # Barabasi Albert random graph
        g_ran = random_type_dict[random_type](n=n, m=int(m / n))
    elif random_type == "KB":
        # Kleinberg model random graph
        lattice_length = math.ceil(math.sqrt(n))
        g_ran = random_type_dict[random_type](n=lattice_length, p=1, q=1, r=10, dim=2)
    elif random_type == "newman_WS":
        # Newmann-Watts-Strogatz random graph
        g_ran = random_type_dict[random_type](n=n, k=int(m / n) * 2, p=0.01)

    g_ran = nx.Graph(g_ran)

    return g_ran


def graph_preprocessing(G):
    """
    This function takes an input graph G and modifies it such that it is undirected,
    has no self-loops, and no disconnected components.
    Parameters
    ----------
    G : graph
        A NetworkX graph object
    Returns
    -------
    G : graph
        A NetworkX graph object which is undirected, has no self-loops, and no disconnected components
    """
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if nx.number_connected_components(G) > 1:
        biggest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(biggest_component)
    if not nx.is_connected(G):
        raise ValueError("Graph is not connected")
    return G


def fit_bass_model(data):
    """
    This function fits the Bass model on the input data. The Bass model is a mathematical
    model that describes the growth of a new idea/product in the society/market.
    Parameters
    ----------
    data : list
        A list of numerical data representing the adoption rate of a new product over time.
    Returns
    -------
    c_t : list
        A list of predicted adoption rate over time.
    """

    def c_t(x, p, q, m):
        return (p + (q / m) * (x)) * (m - x)

    popt, pcov = curve_fit(c_t, xdata=range(len(data)), ydata=data)
    p, q, m = popt[0], popt[1], popt[2]
    return c_t(range(len(data)), p, q, m), p, q, m


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
    - adopters_list (list): list containing the number of adopters after each round of the simulation

    """
    G = graph
    n = len(G.nodes())
    initial_adopt_num = int(percentage_initial_adopt * n)

    if centrality_measure == "degree":
        # sort the nodes by degree
        node_and_degree = G.degree()
        (node, degree) = zip(*node_and_degree)
        sorted_nodes = [x for (y, x) in sorted(zip(degree, node), reverse=True)]
    elif centrality_measure == "pagerank":
        # sort the nodes by pagerank
        pr = nx.pagerank(G, alpha=0.85)
        sorted_nodes = sorted(pr, key=pr.get, reverse=True)
    elif centrality_measure == "closeness":
        # sort the nodes by closeness
        cc = nx.closeness_centrality(G)
        sorted_nodes = sorted(cc, key=cc.get, reverse=True)

    # eigenvector centrality for each node
    ec = nx.eigenvector_centrality(G, tol=1.0e-3)
    # normalize the eigenvector centrality to be between 0 and 1
    ec = {k: v / max(ec.values()) for k, v in ec.items()}

    # add status & threshold attribute to nodes
    for node in G.nodes():
        G.nodes[node]["status"] = 0
        G.nodes[node]["threshold"] = random.random()
        G.nodes[node]["influence"] = ec[node]

    # set initial adopters
    if type_initial_adopt == "central":
        for node in sorted_nodes[:initial_adopt_num]:
            G.nodes[node]["status"] = 1
    elif type_initial_adopt == "marginal":
        for node in sorted_nodes[-initial_adopt_num:]:
            G.nodes[node]["status"] = 1
    else:
        for i in range(initial_adopt_num):
            node = random.choice(list(G.nodes()))
            G.nodes[node]["status"] = 1

    # spread of information
    t = timestamp
    # list of adopters
    adopters_list = [initial_adopt_num]
    for i in range(t - 1):
        # if full diffusion happened then stop the process
        if i != 0 and adopters_list[-1] == n:
            # fill the list with the last value which is the number of nodes
            adopters_list += [n] * (t - len(adopters_list))
            # print(f'Diffusion already stopped at time step: {i+1}')
            # break out of the outer for loop
            break
        # iterate over nodes
        for node in G.nodes():
            # if the node is not an adopter
            if G.nodes[node]["status"] == 0:
                # get neighbors and check if they are adopters, calculate the percentage of neighbors that are adopters
                neighbors = list(G.neighbors(node))
                count = 0
                total_influence = 0
                for neighbor in neighbors:
                    if G.nodes[neighbor]["status"] == 1:
                        count += 1
                        # total influence of adopters, normalized by the number of neighbors
                        total_influence += G.nodes[neighbor]["influence"]
                percentage = count / len(neighbors)
                G.nodes[node]["percentage_adopt"] = percentage
                G.nodes[node]["neighbor_influence"] = total_influence / len(neighbors)

                # calculate the probability of adoption (mean of percentage and threshold)
                if G.nodes[node]["percentage_adopt"] == 0:
                    G.nodes[node]["probability"] = 0
                else:
                    G.nodes[node]["probability"] = (
                        G.nodes[node]["percentage_adopt"]
                        + G.nodes[node]["threshold"]
                        + G.nodes[node]["neighbor_influence"]
                    ) / 3

        # update adoption after first full node iteration, when all probabilities are updated
        for node in G.nodes():
            # if the node is not an adopter
            if G.nodes[node]["status"] == 0:        
                # sample from a uniform distribution and check if the probability is greater than the sample
                if random.random() < G.nodes[node]["probability"]:
                    G.nodes[node]["status"] = 1

        # get list of adopters
        adopters = [node for node in G.nodes() if G.nodes[node]["status"] == 1]
        adopters_list.append(len(adopters))

    return adopters_list


def make_diffusions_and_average(graph, num_simulations, timestamp, percentage_initial_adopt, type_initial_adopt,centrality_measure):
    """
    This function simulates several diffusions for the SAME graph and averages the results
    Parameters
    ----------
    num_simulations : int
        Number of simulations to run
    graph : NetworkX Graph
        The graph on which the simulation is run
    timestamp : int
        Number of time steps for which the simulation will be run
    percentage_initial_adopt : float
        The percentage of nodes that will be selected as initial adopters
    type_initial_adopt : str
        The criterion for selecting initial adopters, either "central", "marginal" or "random"
    centrality_measure : str
        The centrality measure that will be used to select initial adopters
    Returns
    -------
    averaged_diffusion_list : numpy array
        The average diffusion over all the simulations
    """
    cumulated_diffusion_list = list()
    for i in tqdm(range(num_simulations)):
        diff_list = spread_of_information(
            graph,
            timestamp,
            percentage_initial_adopt,
            type_initial_adopt,
            centrality_measure,
        )
        cumulated_diffusion_list.append(diff_list)
    cumulated_diffusion_list = np.array(cumulated_diffusion_list)
    averaged_diffusion_list = np.mean(cumulated_diffusion_list, axis=0)
    return averaged_diffusion_list


def run_simulation(G, num_simulations, time_steps, percentage_initial_adopt, centrality_measure, name):
    """
    Runs the simulation for the given graph and saves the results in a csv file
    Parameters
    ----------
    G : NetworkX Graph
        The graph on which the simulation is run
    num_simulations : int
        Number of simulations to run
    time_steps : int
        Number of time steps for which the simulation will be run
    percentage_initial_adopt : float
        The percentage of nodes that will be selected as initial adopters
    centrality_measure : str
        The centrality measure that will be used to select initial adopters
    name : str
        The name of the graph
    Returns
    -------
    ran_perc, ran_cent, ran_diff : list
    """
    # run the simulation
    ran = make_diffusions_and_average(
        G,
        num_simulations,
        time_steps,
        percentage_initial_adopt,
        "random",
        centrality_measure,
    )
    cen = make_diffusions_and_average(
        G,
        num_simulations,
        time_steps,
        percentage_initial_adopt,
        "central",
        centrality_measure,
    )
    mar = make_diffusions_and_average(
        G,
        num_simulations,
        time_steps,
        percentage_initial_adopt,
        "marginal",
        centrality_measure,
    )

    # get percentage of adopters
    ran_perc = [x / len(G.nodes()) for x in ran]
    cen_perc = [x / len(G.nodes()) for x in cen]
    mar_perc = [x / len(G.nodes()) for x in mar]

    # dataframes for the results, each row is a time step, the three columns are the three types of initial adopters
    df_ran = pd.DataFrame(ran_perc, columns=["random"])
    df_cen = pd.DataFrame(cen_perc, columns=["central"])
    df_mar = pd.DataFrame(mar_perc, columns=["marginal"])
    df = pd.concat([df_ran, df_cen, df_mar], axis=1)

    # create or append results to csv file
    if not os.path.exists(f"../results/{name}"):
        # create a new directory for the graph
        os.makedirs(f"../results/{name}")

    # save the results to a csv file
    df.to_csv(f"../results/{name}/{name}_{centrality_measure}_results.csv")

    return ran_perc, cen_perc, mar_perc


def take_sufficient_index(lists):
    """
    This function returns the index where after the index, none of the lists have any more value changes
    ----------
    lists : list
        List of lists, each list contains a diffusion
    Returns
    -------
    maximum index: int
        index where plot only shows relevant information
    """
    maximum_index = len(lists[0])
    for l in lists:
        old_value = l[-1]
        for index, new_value in enumerate(l[::-1][1:], start=1):
            if new_value != old_value:
                if index < maximum_index:
                    maximum_index = index
                break
    if maximum_index == len(lists[0]) or maximum_index == 1 or maximum_index == 2:
        return len(lists[0])
    else:
        return len(lists[0]) - maximum_index + 2


def plotting_adopters(diff_data, name):
    """
    This function plots for the diffusion for the same graphs different adopter types
    Parameters
    ----------
    diff_data : list
        List of lists, each list contains the diffusion for a different adopter type
    name : str
        The name of the graph
    Returns
    -------
    None
    """

    ran_perc, cen_perc, mar_perc = diff_data

    # crop the lists to relevant t-window
    suf_t = take_sufficient_index([ran_perc, cen_perc, mar_perc])
    print(suf_t)
    ran_perc, cen_perc, mar_perc = ran_perc[:suf_t], cen_perc[:suf_t], mar_perc[:suf_t]

    # plot the number of adopters over time with seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(ran_perc, label="random", color="red")
    plt.plot(cen_perc, label="central", color="green")
    plt.plot(mar_perc, label="marginal", color="blue")
    plt.xlabel("Time steps")
    plt.ylabel("Percentage of adopters")
    plt.title(f"{name} network")
    plt.legend()
    plt.savefig(f"../results/{name}/{name}_diff.png")
    plt.show(block=False)


def plotting_structure(diff_data, name):
    """
    This function plots the diffusion for different graphs
    Parameters
    ----------
    diff_data : list
        List of tuples containing the diffusion data for each graph
    name : str
        The name of the graph

    Returns
    -------
    None
    """
    facebook, er, ws, ba, nw_ws = diff_data

    # crop the lists to relevant t-window
    suf_t = take_sufficient_index([facebook, er, ws, ba, nw_ws])
    print(suf_t)
    facebook, er, ws, ba, nw_ws = (
        facebook[:suf_t],
        er[:suf_t],
        ws[:suf_t],
        ba[:suf_t],
        nw_ws[:suf_t],
    )

    # plot the number of adopters over time with seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(facebook, label="Facebook", color="yellow")
    plt.plot(er, label="ER", color="red")
    plt.plot(ws, label="WS", color="green")
    plt.plot(ba, label="BA", color="blue")
    plt.plot(nw_ws, label="NW_WS", color="black")
    plt.xlabel("Time steps")
    plt.ylabel("Percentage of adopters")
    plt.title(f"{name} initial adopters in different networks")
    plt.legend()
    plt.savefig(f"../results/{name}_diff.png")
    plt.show(block=False)


def plotting_bass_model(perc, name, type_initial_adopt):
    """
    This function plots the Bass model along with the inputted diffusion curve of a new idea/product given the input data.
    Parameters
    ----------
    diff_data : list of lists
        A list of lists, where each list represents the percentage of adopters of the new product over time with different types of initial adopters.
    name : str
        The name of the graph
    Returns
    -------
    None
    """

    # crop the lists to relevant t-window
    suf_t = take_sufficient_index([perc])
    print(suf_t)
    perc = perc[:suf_t]

    # fit bass model and get parameters
    bass_m, p, q, m = fit_bass_model(perc)

    # plot bass model and diffusion curve
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(perc, label=f"{type_initial_adopt}", color="black")
    plt.plot(bass_m, label="Bass model", color="red")
    plt.xlabel("Time")
    plt.ylabel("Percentage of adopters")
    plt.title(f"Fit of Bass model on {name}")
    plt.legend()
    plt.savefig(f"../results/{type_initial_adopt}_bass_model.png")

    print(f"p = {p}, q = {q}, m = {m}")
    plt.show(block=False)
