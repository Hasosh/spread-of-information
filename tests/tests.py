import os
import pytest
import networkx as nx
import pandas as pd
import numpy as np
from src.functions import show_and_save_prop
from src.functions import generate_random_graph
from src.functions import graph_preprocessing
from src.functions import spread_of_information
from src.functions import make_diffusions_and_average
from src.functions import run_simulation

def test_show_and_save_prop():
    G = nx.erdos_renyi_graph(100, 0.1)
    name = "ER_test"
    show_and_save_prop(G, name)
    assert os.path.exists(f"../results/{name}/{name}_properties.csv")
    assert os.path.exists(f"../results/{name}/{name}_degree_distribution.png")
    os.remove(f"../results/{name}/{name}_properties.csv")
    os.remove(f"../results/{name}/{name}_degree_distribution.png")
    os.rmdir(f"../results/{name}")

def test_generate_random_graph():
    # Test case 1: Check if a graph is returned for valid input
    g_ran = generate_random_graph(random_type="ER", nodes=10, edges=20)
    assert isinstance(g_ran, nx.Graph), "The returned object is not a graph"
    assert g_ran.number_of_nodes() == 10, "The number of nodes is not as expected"
    assert g_ran.number_of_edges() == 20, "The number of edges is not as expected"

    # Test case 2: Check if ValueError is raised for invalid random_type
    with pytest.raises(ValueError) as excinfo:
        g_ran = generate_random_graph(random_type="Invalid", nodes=10, edges=20)
    assert str(excinfo.value) == "Invalid random_type: Invalid. Please choose one of ER / WS / BA / KB / newman_WS"

    # Test case 3: Check if AssertionError is raised for missing input
    with pytest.raises(AssertionError) as excinfo:
        g_ran = generate_random_graph(random_type="ER")
    assert str(excinfo.value) == "Please provide valid input for number of nodes and edges"

def test_graph_preprocessing():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,1)])
    G.add_edge(1,1)
    G.add_node(6)
    G_new = graph_preprocessing(G)
    assert G_new.number_of_nodes() == 5
    assert G_new.number_of_edges() == 5
    assert not any([G_new.has_edge(u, u) for u in G_new.nodes()])
    assert nx.is_connected(G_new)

def test_spread_of_information():
    ## Test case 1: Check if the function returns what is expected for random initial adopters and degree as centrality measure
    G = nx.erdos_renyi_graph(10, 0.5)
    timestamp = 5
    percentage_initial_adopt = 0.2
    type_initial_adopt = "random"
    centrality_measure = "degree"
    adopters_list_d = spread_of_information(G, timestamp, percentage_initial_adopt, type_initial_adopt, centrality_measure)

    # Check if the function returns a list
    assert isinstance(adopters_list_d, list)
    # Check if the length of the adopters_list is equal to the timestamp
    assert len(adopters_list_d) == timestamp
    # Check if the first element of the adopters_list is equal to the expected number of initial adopters
    assert adopters_list_d[0] == int(percentage_initial_adopt * len(G.nodes()))

    ## Test case 2: Check if the function returns what is expected for central initial adopters and pagerank as centrality measure
    G = nx.erdos_renyi_graph(10, 0.5)
    timestamp = 5
    percentage_initial_adopt = 0.2
    type_initial_adopt = "central"
    centrality_measure = "pagerank"
    adopters_list_p = spread_of_information(G, timestamp, percentage_initial_adopt, type_initial_adopt,
                                          centrality_measure)

    # Check if the function returns a list
    assert isinstance(adopters_list_p, list)
    # Check if the length of the adopters_list is equal to the timestamp
    assert len(adopters_list_p) == timestamp

    ## Test case 3: Check if the function returns what is expected for maginal initial adopters and closness as centrality measure
    G = nx.erdos_renyi_graph(10, 0.5)
    timestamp = 5
    percentage_initial_adopt = 0.2
    type_initial_adopt = "marginal"
    centrality_measure = "closeness"
    adopters_list_c = spread_of_information(G, timestamp, percentage_initial_adopt, type_initial_adopt,
                                            centrality_measure)

    # Check if the function returns a list
    assert isinstance(adopters_list_c, list)
    # Check if the length of the adopters_list is equal to the timestamp
    assert len(adopters_list_c) == timestamp

def test_make_diffusions_and_average():
    # create a small test graph
    test_graph = nx.complete_graph(5)
    # run the function with test parameters
    result = make_diffusions_and_average(test_graph, num_simulations=5, timestamp=5,
                                         percentage_initial_adopt=0.1, type_initial_adopt='random',
                                         centrality_measure='degree')
    # check that the result is a numpy array
    assert isinstance(result, np.ndarray)
    # check that the length of the result is equal to the timestamp parameter
    assert len(result) == 5
    # check that the values in the result array are between 0 and 1
    assert all(0 <= i <= 1 for i in result)

def test_run_simulation():
    G = nx.erdos_renyi_graph(1000, 0.1) # generate a random graph
    num_simulations = 5
    time_steps = 5
    percentage_initial_adopt = 0.1
    centrality_measure = 'degree'
    name = 'test_graph'
    os.makedirs(f"../results/{name}", exist_ok=True)
    ran_perc, cen_perc, mar_perc = run_simulation(G, num_simulations, time_steps, percentage_initial_adopt, centrality_measure, name)
    # check if the returned values are lists of the same length
    assert len(ran_perc) == len(cen_perc) == len(mar_perc) == time_steps
    # check if the csv file was created
    assert os.path.exists(f"../results/{name}/{name}_{centrality_measure}_results.csv")
    # check if the values in the csv file are the same as the returned values
    df = pd.read_csv(f"../results/{name}/{name}_{centrality_measure}_results.csv")
    assert np.allclose(df['random'], ran_perc)
    assert np.allclose(df['central'], cen_perc)
    assert np.allclose(df['marginal'], mar_perc)

