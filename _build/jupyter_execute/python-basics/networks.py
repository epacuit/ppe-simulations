# Networks

Many agent-based models assume that agents are located on a **network**, also called a **graph** or **digraph**.   A network is a pair $(N, E)$ where $N\neq \varnothing$ is a non-empty set of **nodes**, and $E\subseteq N\times N$ is a set of ordered pairs of elements from $N$ called the set of **edges**.  

If $n,n'\in N$ and $(n,n')$, then we say that there is an edge from $n$ to $n'$.  

A network $(N, E)$ is **undirected** if for all $n,n'\in N$, if $(n,n')\in E$ then $(n', n)\in E$.  Otherwise, the network is **directed**.  

[Networkx](https://networkx.org/documentation/stable/index.html) is an excellent Python package for the creation, manipulation, and study of networks.

import networkx as nx

G = nx.Graph() # create an undirected graph instance
G.add_nodes_from([0, 1, 2, 3, 4, 5]) # add the nodes from a list
G.add_edges_from([(0,1), (0,2), (5, 1), (1,2), (2,3), (3,4), (3,5), (4,5)]) # add the edges from a list 

# draw the graph
nx.draw(G, pos = nx.circular_layout(G), with_labels = True, font_color="white", font_size=14)


The neighborhors of a node $n$ are the nodes that are directly connected to $n$. That is, in a network $(N, E)$ the neighbors of a node $n\in N$ is the set $\{n'\mid (n,n')\in E\}$.  The neighbors of each of the nodes in the above graph are: 

for n in G.nodes: 
    print(f"The neighbors of {n} are {list(G.neighbors(n))}")


```{note}
Since $G$ is an undirected, we have that $0$ is a neighbor of $1$ even though we only added a single edge $(0,1)$ to the set of edges.   This is different if $G$ was a directed graph. 
```

directed_G = nx.DiGraph() # create an undirected graph instance
directed_G.add_nodes_from([0, 1, 2, 3, 4, 5]) # add the nodes from a list
directed_G.add_edges_from([(0,1), (0,2), (5, 1), (1,2), (2,3), (3,4), (3,5), (4,5)]) # add the edges from a list 
nx.draw(directed_G, pos = nx.circular_layout(directed_G), with_labels = True, font_color="white", font_size=14)
for n in directed_G.nodes: 
    print(f"The neighbors of {n} are {list(directed_G.neighbors(n))}")


The netwrokx package has many algorithms for analyzing networks.  For example, there are methods to calculate the **density** of a network (a measure of how many edges are in a graph) and the **diameter** of a graph (the maximum number of edges between any two nodes).    Since the diameter is only defined for connected graphs, we should use the `is_connected` method to test of the graph is connected. 

See [https://networkx.org/documentation/stable/reference/algorithms/index.html](https://networkx.org/documentation/stable/reference/algorithms/index.html) for an overview of the available functions. 



G = nx.Graph() # create an undirected graph instance
G.add_nodes_from([0, 1, 2, 3, 4, 5]) # add the nodes from a list
G.add_edges_from([(0,1), (0,2), (5, 1), (1,2), (2,3), (3,4), (3,5), (4,5)]) # add the edges from a list 

print("The density of the network is ", nx.density(G))
print("The diameter of the network is ", nx.diameter(G))


## Graph Generators

There are a number of graph generators that make it easy to create networks: [https://networkx.org/documentation/stable/reference/generators.html](https://networkx.org/documentation/stable/reference/generators.html)

num_nodes = 10
G = nx.cycle_graph(num_nodes)

nx.draw(G, pos=nx.circular_layout(G), 
        with_labels = True, 
        font_color="white", 
        font_size=14,
        font_weight='bold')

print("The density of the network is ", nx.density(G))
if nx.is_connected(G):
    print("The diameter of the network is ", nx.diameter(G))
else:
    print("The network is not connected")


num_nodes = 10
G = nx.path_graph(num_nodes)

nx.draw(G, pos=nx.circular_layout(G), 
        with_labels = True, 
        font_color="white", 
        font_size=14,
        font_weight='bold')

print("The density of the network is ", nx.density(G))
if nx.is_connected(G):
    print("The diameter of the network is ", nx.diameter(G))
else:
    print("The network is not connected")


num_nodes = 10
G = nx.wheel_graph(num_nodes)

nx.draw(G, pos=nx.circular_layout(G), 
        with_labels = True, 
        font_color="white", 
        font_size=14,
        font_weight='bold')

print("The density of the network is ", nx.density(G))
if nx.is_connected(G):
    print("The diameter of the network is ", nx.diameter(G))
else:
    print("The network is not connected")


num_nodes = 10
G = nx.empty_graph(num_nodes)

nx.draw(G, pos=nx.circular_layout(G), 
        with_labels = True, 
        font_color="white", 
        font_size=14,
        font_weight='bold')

print("The density of the network is ", nx.density(G))
if nx.is_connected(G):
    print("The diameter of the network is ", nx.diameter(G))
else:
    print("The network is not connected")


num_nodes = 10
G = nx.complete_graph(num_nodes)

nx.draw(G, pos=nx.circular_layout(G), 
        with_labels = True, 
        font_color="white", 
        font_size=14,
        font_weight='bold')

print("The density of the network is ", nx.density(G))
if nx.is_connected(G):
    print("The diameter of the network is ", nx.diameter(G))
else:
    print("The network is not connected")


num_nodes = 10
edge_prob = 0.25

# generate a random graph with the probability  edge_prob of an edge connecting two nodes
G = nx.erdos_renyi_graph(num_nodes, edge_prob)

nx.draw(G, pos=nx.circular_layout(G), 
        with_labels = True, 
        font_color="white", 
        font_size=14,
        font_weight='bold')

print("The density of the network is ", nx.density(G))
if nx.is_connected(G):
    print("The diameter of the network is ", nx.diameter(G))
else:
    print("The network is not connected")
