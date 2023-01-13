## This project aims to study the diffusion of innovation on social networks. The project includes several functions for the following purposes:

- Generating random graphs with different models (Erdos-Renyi, Watts-Strogatz, Barabasi-Albert, and Newman-Watts)
- Preprocessing graphs
- Running simulations on graphs with different types of initial adopters (random, central, and marginal)
- Plotting the results of the simulations
- Saving the results in csv files
- Averaging the results of multiple runs of the simulation

The project also includes a main function that runs the simulation on a real graph and multiple instances of the random graphs and saves the results in a dictionary.

To run the simulation, you need to have the following python libraries installed:

- NetworkX
- NumPy
- Pandas
- Matplotlib
- Seaborn

You can install them by running the following command:
pip install -r requirements.txt

## How to run the simulation

To run the simulation, you need to run the main function in the main.py file. The function takes several parameters:

- num_simulations: the number of times the simulation will be run on each graph.
- num_random_graphs: the number of instances of each random graph that will be generated.
- time_steps: the number of time steps for which the simulation will be run.
- percentage_initial_adopt: the percentage of nodes that will be selected as initial adopters.
- centrality_measures: the centrality measures that will be used to select the initial adopters.
- name: the name of the graph that will be used in the results.

You can change the values of these parameters according to your needs.

## Results

The results of the simulation will be saved in the results directory in the form of .csv files, one file for each graph and centrality measure. Additionally, the simulation results will be plotted in .png format and will be saved in the same directory.

The results of the simulation will also be saved in a dictionary, which includes the average results of the multiple runs of the simulation on each graph. The dictionary can be used for further analysis and comparison of the results.

**Note**
The code runs on csv file with edge list, you can use your own dataset and make the necessary changes in the code.



