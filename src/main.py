from functions import *

def main():
    # Create a graph from edge list
    G = nx.read_edgelist('../data/facebook_combined.txt', delimiter=' ')

    # preprocess the graph
    G = graph_preprocessing(G)

    # show properties
    show_and_save_prop(G, 'facebook')

    # simulation parameters
    num_simulations = 5
    time_steps = 20
    percentage_initial_adopt = 0.025

    # run the simulation
    run_simulation_and_plot(G, num_simulations, time_steps, percentage_initial_adopt, 'degree', 'facebook')


if __name__ == '__main__':
    main()


