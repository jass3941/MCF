# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:18:06 2021

@author: daham.kim
"""


def graph_building(sim_matrix_df, item, scen):
    import networkx as nx
    df_test = sim_matrix_df.copy()
    """
    similarity  = 1 --> normal   -->  do not connect  --> temp = 0
    sililarity <> 1 --> abnormal --> connect --> temp = 
    convert to 0-1 matrix in df_test and return it

    Returns 
            network G
    """
    for i in range(sim_matrix_df.shape[0]):
        for j in range(sim_matrix_df.shape[1]):
            if (df_test.iloc[i, j] == 1):
                temp = 0
            else:
                temp = 1
            df_test.iloc[i, j] = temp

    # converting df_test to graph using networkX
    Graph = nx.DiGraph()
    # list to save cf_item node with attribute green
    cf_item = []
    for it in item:
        cf_item.append((it, {"color": "green"}))
    Graph.add_nodes_from(cf_item)

    # list to save scenario node with attribute red
    scen_index = []
    for sc in scen.iloc[:, 6]:
        scen_index.append((sc, {"color": "red"}))
    Graph.add_nodes_from(scen_index)

    # print nodes generated in graph G
    print(Graph.nodes())

    # adding edges in graph G
    # connect if the scenario(node red) shows abnormal behaviors in certain cf_items(node green)
    # connection only available among scenario(red node) and cf_items(green node)
    # edge adding syntax will be columns of df_test(cf_itmes) and index of df_test(scenario)
    for i in range(df_test.shape[0]):
        for j in range(df_test.shape[1]):
            if df_test.iloc[i, j] == 1:
                Graph.add_edge(df_test.columns[j], df_test.index[i])  # j to i directed graph
    # print edge of graph G
    # print(Graph.edges.data())

    # calcuate of degree of nodes
    # degree measures how much connection each scenario has
    # meaning that how many abnormal cashflow items each scenario have
    # store this info for further use --> not used now
    degree = dict(Graph.degree())

    return Graph


def graph_visualization(Graph):
    import networkx as nx
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(Graph, seed=1000, k=1)
    plt.figure(figsize=(60, 60))  # Fix figure size
    nx.draw_networkx(Graph, pos,
                     with_labels=True,
                     font_size=3,
                     width=1)


# from itertools import product

def simrank_similarity(graph, source, target, importance_factor=0.9, max_iterations=1000, tolerance=1e-4):
    print("source is : ", source, " -> ",  "  target is : ", target)
    prev_sim = None
    # build up our similarity adjacency dictionary output
    new_sim = {u: {v: 1 if u == v else 0 for v in graph} for u in graph}

    # These functions compute the update to the similarity value of the nodes
    # `u` and `v` with respect to the previous similarity values.
    def avg_sim(s):
        return sum(new_sim[w][x] for (w, x) in s) / len(s) if s else 0.0

    def sim(u, v):
        Gadj = graph.pred if graph.is_directed() else graph.adj
        return importance_factor * avg_sim(list(product(Gadj[u], Gadj[v])))

    for _ in range(max_iterations):
        if prev_sim and _is_close(prev_sim, new_sim, tolerance):
            break
        prev_sim = new_sim
        new_sim = {
            u: {v: sim(u, v) if u is not v else 1 for v in new_sim[u]} for u in new_sim
        }

    if source is not None and target is not None:
        return new_sim[source][target]
    if source is not None:
        return new_sim[source]

    return new_sim


def _is_close(d1, d2, atolerance=0, rtolerance=0):
    """Determines whether two adjacency matrices are within a provided tolerance.
    d1 : dict(Adjacency dictionary)
    d2 : dict(Adjacency dictionary)
    atolerance : Some scalar tolerance value to determine closeness
    rtolerance : A scalar tolerance value that will be some proportion of ``d2``'s value

    Returns : boolean close or not
    """
    # Pre-condition: d1 and d2 have the same keys at each level if they
    # are dictionaries.
    if not isinstance(d1, dict) and not isinstance(d2, dict):
        return abs(d1 - d2) <= atolerance + rtolerance * abs(d2)
    return all(all(_is_close(d1[u][v], d2[u][v]) for v in d1[u]) for u in d1)


def product(*args, repeat=1):
    # source from https://docs.python.org/3/library/itertools.html#itertools.product
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
