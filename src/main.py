"""
This script runs a diffusion simulation on a real graph and several random graphs, comparing the results of different
types of initial adopters.
The user can specify the number of simulations to run, the number of time steps for each simulation, and the percentage
of initial adopters.
It also saves the results of various graph properties and the simulation results in .csv files and plots.
"""
from functions import *
import time
import matplotlib.pyplot as plt

def main():
    start = time.time()

    # simulation parameters
    num_simulations = 5
    num_random_graphs = 5
    time_steps = 20
    percentage_initial_adopt = 0.01
    centrality_measures = 'degree' # 'pagerank' or 'clossness'
    name = 'facebook_politician'

    averaged_results = dict()

    # real graph
    # Create a graph from edge list
    G_fb = nx.read_edgelist(f'../data/{name}_edges.csv', delimiter=',') # G_fb = nx.read_edgelist('../data/facebook_combined.txt', delimiter=' ')

    # preprocess the graph
    G_fb = graph_preprocessing(G_fb)

    # show properties
    show_and_save_prop(G_fb, name)

    # run the simulation
    ran_avg, cen_avg, mar_avg = run_simulation(G_fb, num_simulations, time_steps, percentage_initial_adopt,
                                               centrality_measures, name)

    print('Real graph finished')
    # save results
    averaged_results[name] = (ran_avg, cen_avg, mar_avg)

    # random graphs
    collected_results = dict()
    for i in range(num_random_graphs):
        # create random graphs
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

        # run the simulation
        er_ran_perc, er_cen_perc, er_mar_perc = run_simulation(G_er, num_simulations, time_steps,
                                                               percentage_initial_adopt, centrality_measures, 'ER')
        ws_ran_perc, ws_cen_perc, ws_mar_perc = run_simulation(G_ws, num_simulations, time_steps,
                                                               percentage_initial_adopt, centrality_measures, 'WS')
        ba_ran_perc, ba_cen_perc, ba_mar_perc = run_simulation(G_ba, num_simulations, time_steps,
                                                               percentage_initial_adopt, centrality_measures, 'BA')
        nw_ws_ran_perc, nw_ws_cen_perc, nw_ws_mar_perc = run_simulation(G_nw_ws, num_simulations, time_steps,
                                                                        percentage_initial_adopt, centrality_measures,
                                                                        'newman_WS')


        # save results in dictionary
        if i == 0:
            collected_results["ER"] = [(er_ran_perc, er_cen_perc, er_mar_perc)]
            collected_results["WS"] = [(ws_ran_perc, ws_cen_perc, ws_mar_perc)]
            collected_results["BA"] = [(ba_ran_perc, ba_cen_perc, ba_mar_perc)]
            collected_results["newman_WS"] = [(nw_ws_ran_perc, nw_ws_cen_perc, nw_ws_mar_perc)]
        else:
            collected_results["ER"].append((er_ran_perc, er_cen_perc, er_mar_perc))
            collected_results["WS"].append((ws_ran_perc, ws_cen_perc, ws_mar_perc))
            collected_results["BA"].append((ba_ran_perc, ba_cen_perc, ba_mar_perc))
            collected_results["newman_WS"].append((nw_ws_ran_perc, nw_ws_cen_perc, nw_ws_mar_perc))

    # average the results
    for key in collected_results:
        ran_avg = np.mean([res[0] for res in collected_results[key]], axis=0)
        cen_avg = np.mean([res[1] for res in collected_results[key]], axis=0)
        mar_avg = np.mean([res[2] for res in collected_results[key]], axis=0)
        averaged_results[key] = (ran_avg, cen_avg, mar_avg)

    # delete collected results for memory saving
    collected_results.clear()

    # convert averaged results to dataframe and save to csv
    df = pd.DataFrame.from_dict(averaged_results, orient='index')
    # rename columns
    df.columns = ['random', 'centrality', 'marginal']
    df.to_csv(f'../results/all_results_{name}.csv')

    print('Random graphs finished')
    # plotting the results
    # plotting same graph different adopters
    for key in averaged_results:
        plotting_adopters(averaged_results[key], key)

    # plotting different graph same adopters
    plotting_structure((averaged_results[name][0], averaged_results["ER"][0], averaged_results["WS"][0]
                        , averaged_results["BA"][0], averaged_results["newman_WS"][0]), "random")
    plotting_structure((averaged_results[name][1], averaged_results["ER"][1], averaged_results["WS"][1]
                        , averaged_results["BA"][1], averaged_results["newman_WS"][1]), "central")
    plotting_structure((averaged_results[name][2], averaged_results["ER"][2], averaged_results["WS"][2]
                        , averaged_results["BA"][2], averaged_results["newman_WS"][2]), "marginal")

    # plotting bass model on facebook network
    plotting_bass_model(averaged_results[name][0], name, "random")
    plotting_bass_model(averaged_results[name][1], name, "central")
    plotting_bass_model(averaged_results[name][2], name, "marginal")

    # show all plots at once at the end
    plt.show()

    end = time.time()
    print("Time taken: ", (end - start)/60, " minutes")

if __name__ == '__main__':
    main()



