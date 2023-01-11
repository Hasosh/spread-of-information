library(ggplot2)
library(igraph)

setwd("~/Downloads")

set.seed(4)

# Analyze the properties of a Facebook social network graph, including measures of graph properties and a plot of the degree distribution
generate_graph <- function(file, dir=FALSE) {
  
  # Read in the text/csv file and create a graph object
  el <- read.table(file, sep = ' ')
  g <- graph_from_data_frame(el, directed=dir)
  
  
  # Print the measures of graph properties
  print(paste("Number of edges:", ecount(g)))
  print(paste("Number of nodes:", vcount(g)))
  print(paste("Clustering coefficient:", transitivity(g)))
  print(paste("Number of triangles:", sum(count_triangles(g))))
  print(paste("Graph density:", edge_density(g, loops=FALSE)))
  print(paste("Average path lenght:", average.path.length(g)))
  print(paste("Diameter:", diameter(g)))
  
  # Create a histogram of the degree distribution
  degrees <- degree(g)
  hist(degrees, main="Degree Distribution", xlab="Degree", ylab="Frequency")
  
  # Create a density plot of the degree distribution on a log-log scale
  df <- data.frame(degrees)
  plot <- ggplot(df, aes(x=degrees)) + 
    geom_density() + 
    scale_x_log10() + 
    scale_y_log10() + 
    labs(title="Degree Distribution (Log-Log Scale)", x="Degree (Log)", y="Frequency (Log)")
  
  print(plot)
  
  # Return the adjacency matrix
  return(g)
}

# Generates a random graph
generate_random_graph <- function(type, g){
  if (type == "ER"){
    # Erdös Renyi random graph
    g_ran <- sample_gnm(vcount(g), ecount(g), directed = FALSE)
  } else if (type == "WS"){
    # Watts Strogatz random graph
    g_ran <- sample_smallworld(dim = 1, size = vcount(g), nei = floor(ecount(g)/vcount(g)), p = 0.01) # rewiring with p = 0.01: high clustering but low diameter
  } else if (type == "BA"){
    # Barabasi Albert random graph
    g_ran <- sample_pa(vcount(g), power = 1, m = floor(ecount(g)/vcount(g)), directed = FALSE)
  }
  return(g_ran)
}

# Analyze the properties of a random graph, including measures of graph properties and a plot of the degree distribution
show_prop <- function(g) {
  
  # Print the measures of graph properties
  print(paste("Number of edges:", ecount(g)))
  print(paste("Number of nodes:", vcount(g)))
  print(paste("Clustering coefficient:", transitivity(g)))
  print(paste("Number of triangles:", sum(count_triangles(g))))
  print(paste("Graph density:", edge_density(g, loops=FALSE)))
  print(paste("Average path lenght:", average.path.length(g)))
  print(paste("Diameter:", diameter(g)))
  
  
  # Create a histogram of the degree distribution
  degrees <- degree(g)
  hist(degrees, main="Degree Distribution", xlab="Degree", ylab="Frequency")
  
  # Create a density plot of the degree distribution on a log-log scale
  df <- data.frame(degrees)
  plot <- ggplot(df, aes(x=degrees)) + 
    geom_density() + 
    scale_x_log10() + 
    scale_y_log10() + 
    labs(title="Degree Distribution (Log-Log Scale)", x="Degree (Log)", y="Frequency (Log)")
  
  print(plot)
  
  # Return the adjacency matrix
  return(g)
}

# Real network
g_real <- generate_graph("facebook_combined.txt")


spread_of_information <- function(graph, time, percent_initial_adop, type_initial_adop){
  g<- graph
  vertices <- vcount(g)
  initial_adop <- as.integer(vertices*percent_initial_adop)

  df <- data.frame(vertex=as_ids(V(g)), degree=degree(g)) 
  
  if (type_initial_adop == "central"){
    df <- df[order(-df$degree), ]
    nodes_degree_sorted = as.integer(df$vertex)
    seed <- nodes_degree_sorted[1:initial_adop]
  } else if (type_initial_adop == "marginal"){
    df <- df[order(df$degree), ]
    nodes_degree_sorted = as.integer(df$vertex)
    seed <- nodes_degree_sorted[1:initial_adop]
  } else if (type_initial_adop == "random"){
    seed <- sample(V(g),initial_adop)
  }

  
  V(g)$threshold = runif(vcount(g), 0.01, 0.99)
  V(g)$status=0 # Create a vertex attribute for adoption status. 1 if the node has adopted the innovation. 0 if not.
  V(g)$status[seed]=1 #These 'seed' individuals get a status of 1 at the beginning.
  #l=layout_with_fr(g)
  #plot(g, vertex.label="", vertex.size=8, vertex.color=c("darkgray", "red")[V(g)$status+1], layout=l)
  

   t = time #time steps to run the simulation
   g.time=list() #empty list to store the output networks
   g.time[[1]]=g
   for(j in 1:t){
     # Get the indices of the nodes that have status = 0
     non_adopters <- which(V(g)$status == 0)
     
     # Calculate the number of adopting neighbors for the non-adopting nodes
     V(g)$adopting_neighbors[non_adopters] <- sapply(V(g)[non_adopters], function(x) sum(V(g)$status[neighbors(g,x)]))
       
     # Calculate the percentage of adopting neighbors for the non-adopting nodes
     V(g)$percent_neighbors_adopted[non_adopters] <- V(g)$adopting_neighbors[non_adopters] / sapply(V(g)[non_adopters], function(x) length(neighbors(g,x)))

    # Calculate p for non-adopter nodes
    V(g)$p[non_adopters][(V(g)$percent_neighbors_adopted[non_adopters]) == 0] <- 0.0
    V(g)$p[non_adopters][(V(g)$percent_neighbors_adopted[non_adopters]) != 0] <- ((V(g)$percent_neighbors_adopted[non_adopters]*0.4 + V(g)$threshold[non_adopters]*0.6) / 2)
    V(g)$p[non_adopters][(V(g)$p[non_adopters]) >= 1] <- 0.99
       
    adopters=sapply(V(g)$p[non_adopters], function(x) sample(c(1,0), 1, prob=c(x, 1-x)))
       
    # Set the status of the nodes that adopted the new information to 1
    V(g)$status[non_adopters] <- adopters
    #print(V(g)$status[non_adopters])
    # Add the current network to the list of output networks
    g.time[[j+1]]=g
  }

    n.adopt.social=sapply(g.time, function(x) length(which(V(x)$status==1))) #for each time step, count the number of adopters.
    return(n.adopt.social)
}


g<- g_real
cen <- spread_of_information(g, 30, 0.025, "central")
ran <- spread_of_information(g, 30, 0.025, "random")
mar <- spread_of_information(g, 30, 0.025, "marginal")

cen_norm <- cen/vcount(g)
ran_norm <- ran/vcount(g)
mar_norm <- mar/vcount(g)

plot(cen_norm, type = "b", xlab="Time", ylab="Cumulative number of nodes adopted", main = "Spread of Information", ylim = c(0,1), col="black")
points(ran_norm, type = "b", col="red", lty=2)
points(mar_norm, type = "b", col= "blue", lty=3)
legend(1, legend=c("central", "random", "marginal"), col = c("black", "red", "blue"), lty=1:3, cex=0.8)





