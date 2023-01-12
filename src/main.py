from functions import *

def main():
    # facebook
    # Create a graph from edge list
    G_fb = nx.read_edgelist('../data/facebook_combined.txt', delimiter=' ')

    # preprocess the graph
    G_fb = graph_preprocessing(G_fb)

    # show properties
    show_and_save_prop(G_fb, 'facebook')

    # random graphs
    G_er = generate_random_graph(G_fb, 'ER')
    G_ws = generate_random_graph(G_fb, 'WS')
    G_ba = generate_random_graph(G_fb, 'BA')
    G_nw_ws = generate_random_graph(G_fb, 'newman_WS')

    # preprocess the graphs
    G_er = graph_preprocessing(G_er)
    G_ws = graph_preprocessing(G_ws)
    G_ba = graph_preprocessing(G_ba)
    G_nw_ws = graph_preprocessing(G_nw_ws)

    # show properties
    show_and_save_prop(G_er, 'ER')
    show_and_save_prop(G_ws, 'WS')
    show_and_save_prop(G_ba, 'BA')
    show_and_save_prop(G_nw_ws, 'newman_WS')


    # simulation parameters
    num_simulations = 5
    time_steps = 20
    percentage_initial_adopt = 0.025

    # run the simulation
    run_simulation_and_plot(G_fb, num_simulations, time_steps, percentage_initial_adopt, 'degree', 'facebook')
    run_simulation_and_plot(G_er, num_simulations, time_steps, percentage_initial_adopt, 'degree', 'ER')
    run_simulation_and_plot(G_ws, num_simulations, time_steps, percentage_initial_adopt, 'degree', 'WS')
    run_simulation_and_plot(G_ba, num_simulations, time_steps, percentage_initial_adopt, 'degree', 'BA')
    run_simulation_and_plot(G_nw_ws, num_simulations, time_steps, percentage_initial_adopt, 'degree', 'newman_WS')

if __name__ == '__main__':
    main()


