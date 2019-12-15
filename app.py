#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from textwrap import dedent as d
import random

# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

server = app.server

DEBUG_MODE = False

DEPTH = 0
GRAPH_LAYOUT = "top_down"
NODE_LAYOUT = "id"

# raw_edges = pd.read_csv('dataset/pet_supplies_edges.csv')
# raw_nodes = pd.read_csv('dataset/pet_supplies.csv')
# DEFAULT_CATEGORY = "Flea & Tick Center" ## Pet Supplies

raw_edges = pd.read_csv('dataset/musical_instruments_edges.csv')
raw_nodes = pd.read_csv('dataset/musical_instruments.csv')
DEFAULT_CATEGORY = "Instrument Accessories" ## Musical Instruments

# raw_edges = pd.read_csv('dataset/books_edges.csv')
# raw_nodes = pd.read_csv('dataset/books.csv')
# DEFAULT_CATEGORY = "Reference" ## Books

# Source for hierarchy positioning of nodes: https://stackoverflow.com/a/29597209/5404805
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

##############################################################################################################################################################
def network_graph(graphDepth, CategoryToSearch, graphLayout, nodeLayout):

    if not CategoryToSearch:
        CategoryToSearch = DEFAULT_CATEGORY

    node1 = raw_nodes
    input_ids = raw_nodes[raw_nodes['name'] == CategoryToSearch]['id']
    input_id = input_ids.iloc[0]  # TODO: Spit this into multiple subgraphs?

    if DEBUG_MODE:
        print("")
        print("*********************************************************")
        print("input_ids = ")
        print(input_ids)
        print("input_id = ")
        print(input_id)
        print("*********************************************************")
        print("")

    edge1 = raw_edges[raw_edges['src'] == int(input_id)]

    ### Commenting out this segment
    # # filter the record by datetime, to enable interactive control through the input box
    # edge1['Datetime'] = "" # add empty Datetime column to edge1 dataframe
    # categorySet = set()  # contain unique account
    # for index in range(0, len(edge1)):
    #     edge1['Datetime'][index] = datetime.strptime(edge1['Date'][index], '%d/%m/%Y')
    #     if edge1['Datetime'][index].year<yearRange[0] or edge1['Datetime'][index].year>yearRange[1]:
    #         edge1.drop(axis=0, index=index, inplace=True)
    #         continue
    #     categorySet.add(edge1['src'][index])
    #     categorySet.add(edge1['dest'][index])

    # graphDepth level filtering
    edges_to_append = pd.DataFrame()
    for i in range(1, graphDepth + 1):
        if edges_to_append.empty and i == 1:
            curr_edges = edge1
        else:
            curr_edges = edges_to_append
            edges_to_append = pd.DataFrame()
        for index, row in curr_edges.iterrows():
            edges_to_append = edges_to_append.append(raw_edges[raw_edges['src'] == curr_edges['dest'][index]])
        edge1 = edge1.append(edges_to_append)

    if DEBUG_MODE:
        print("")
        print("*********************************************************")
        print("Graph Depth = ")
        print(graphDepth)
        print(edge1)
        print("*********************************************************")
        print("")

    categorySet = set()  # contain unique categories
    for index, row in edge1.iterrows():
        categorySet.add(edge1['src'][index])
        categorySet.add(edge1['dest'][index])

    # to define the centric point of the networkx layout
    shells = []
    shell1 = []
    shell1.append(CategoryToSearch)
    shells.append(shell1)
    shell2 = []
    for ele in categorySet:
        if ele != CategoryToSearch:
            shell2.append(ele)
    shells.append(shell2)

    G = nx.from_pandas_edgelist(edge1, 'src', 'dest', ['src', 'dest'], create_using=nx.MultiDiGraph())
    nx.set_node_attributes(G, node1.set_index('id')['name'].to_dict(), 'CategoryName')
    nx.set_node_attributes(G, node1.set_index('id')['productCount'].to_dict(), 'ProductCount')
    nx.set_node_attributes(G, node1['id'].to_dict(), 'Id')

    # nx.layout.shell_layout only works for more than 3 nodes
    if len(shell2) > 1:
        if graphLayout == "circular":
            pos = nx.layout.shell_layout(G, shells)
        if graphLayout == "top_down":
            pos = hierarchy_pos(G, input_id)
    else:
        pos = nx.layout.spring_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    if len(shell2) == 0:
        traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

        node_trace = go.Scatter(x=tuple([1]), y=tuple([1]), text=tuple([str(CategoryToSearch)]),
                                textposition="bottom center",
                                mode='markers+text',
                                marker={'size': 50, 'color': 'LightSkyBlue'})
        traceRecode.append(node_trace)

        node_trace1 = go.Scatter(x=tuple([1]), y=tuple([1]),
                                 mode='markers',
                                 marker={'size': 50, 'color': 'LightSkyBlue'},
                                 opacity=0)
        traceRecode.append(node_trace1)

        figure = {
            "data": traceRecode,
            "layout": go.Layout(title='Interactive Transaction Visualization', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return figure

    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

    ############################################################################################################################################################

    colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
    colors = ['rgb' + str(x.rgb) for x in colors]

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        # TODO: Add weight as per the number of children per edge
        # weight = float(G.edges[edge]['TransactionAmt']) / max(edge1['TransactionAmt']) * 10
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           line={'width': 1},
                           marker=dict(color=colors[index]),
                           line_shape='spline',
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1

    ###############################################################################################################################################################

    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 20, 'color': 'LightSkyBlue'})

    index = 0

    if DEBUG_MODE:
        for node in G.nodes():
            print(G.nodes[node])

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = "CategoryName: " + str(G.nodes[node]['CategoryName']) + "<br>" + "ProductCount: " + str(
            G.nodes[node]['ProductCount'])
        text = ""
        if nodeLayout == 'name':
            text = str(G.nodes[node]['CategoryName'])
        if nodeLayout == 'id':
            text = str(G.nodes[node]['Id'])
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
        index = index + 1

    traceRecode.append(node_trace)

    ################################################################################################################################################################

    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'},
                                    opacity=0)

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['src']) + "<br>" + "To: " + str(
            G.edges[edge]['dest'])
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)

    #################################################################################################################################################################

    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Interactive Transaction Visualization', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600,
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x',
                                    ayref='y',
                                    x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x',
                                    yref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=2,
                                    arrowwidth=0.2,
                                    opacity=1
                                ) for edge in G.edges]
                            )}
    return figure


######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    #########################Title
    html.Div([html.H1("Transaction Network Graph")],
             className="row",
             style={'textAlign': "center"}),
    #############################################################################################define the row
    html.Div(
        className="row",
        children=[
            ##############################################left side two input components
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            **Graph Depth**

                            Slide the bar to define Graph Depth.
                            """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Slider(
                                id='my-slider',
                                min=0,
                                max=10,
                                step=1,
                                value=DEPTH,
                                marks={
                                    0: {'label': '0'},
                                    1: {'label': '1'},
                                    2: {'label': '2'},
                                    3: {'label': '3'},
                                    4: {'label': '4'},
                                    5: {'label': '5'},
                                    6: {'label': '6'},
                                    7: {'label': '7'},
                                    8: {'label': '8'},
                                    9: {'label': '9'},
                                    10: {'label': '10'}
                                }
                            ),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                        style={'height': '300px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **Category**

                            Input the root category to visualize.
                            """)),
                            dcc.Input(id="input1", type="text", placeholder="Category"),
                            html.Div(id="output")
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),

            ############################################middle graph component
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_graph(DEPTH, DEFAULT_CATEGORY, GRAPH_LAYOUT, NODE_LAYOUT))],
            ),

            #########################################right side two input component
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                    **Graph Layout**
                    Select how the graph looks.
                    """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.RadioItems(
                                id="graph-layout",
                                options=[
                                    {'label': 'Cytoscape', 'value': 'circular'},
                                    {'label': 'Top Down', 'value': 'top_down'}
                                ],
                                value='top_down'
                            )
                        ],
                        style={'height': '300px'}
                    ),
                    dcc.Markdown(d("""
                    **Node Layout**
                    Select how the nodes look.
                    """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.RadioItems(
                                id="node-layout",
                                options=[
                                    {'label': 'Id', 'value': 'id'},
                                    {'label': 'Name', 'value': 'name'}
                                ],
                                value='id'
                            )
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),
        ]
    )
])

################################### callbacks for all components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value'),
     dash.dependencies.Input('input1', 'value'),
     dash.dependencies.Input('graph-layout', 'value'),
     dash.dependencies.Input('node-layout', 'value')])
def update_output(value, input1, graph_layout, node_layout):
    return network_graph(value, input1, graph_layout, node_layout)

if __name__ == '__main__':
    app.run_server(debug=False)
