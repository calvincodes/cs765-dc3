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
import json

import os.path

# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

server = app.server

DEPTH = 0
DEBUG_MODE = False

# raw_edges = pd.read_csv(os.path.dirname(__file__) + 'dataset/pet_supplies_edges.csv')
# raw_nodes = pd.read_csv(os.path.dirname(__file__) + 'dataset/pet_supplies.csv')
# DEFAULT_CATEGORY = "Flea & Tick Center" ## Pet Supplies

raw_edges = pd.read_csv(os.path.dirname(__file__) + 'dataset/musical_instruments_edges.csv')
raw_nodes = pd.read_csv(os.path.dirname(__file__) + 'dataset/musical_instruments.csv')
DEFAULT_CATEGORY = "Instrument Accessories" ## Musical Instruments

# raw_edges = pd.read_csv(os.path.dirname(__file__) + 'dataset/books_edges.csv')
# raw_nodes = pd.read_csv(os.path.dirname(__file__) + 'dataset/books.csv')
# DEFAULT_CATEGORY = "Reference" ## Books

##############################################################################################################################################################
def network_graph(graphDepth, CategoryToSearch):

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

    # nx.layout.shell_layout only works for more than 3 nodes
    if len(shell2) > 1:
        pos = nx.layout.shell_layout(G, shells)
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
                            hoverinfo="text", marker={'size': 50, 'color': 'LightSkyBlue'})

    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = "CategoryName: " + str(G.nodes[node]['CategoryName']) + "<br>" + "ProductCount: " + str(
            G.nodes[node]['ProductCount'])
        text = str(G.nodes[node]['CategoryName'])
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
                                    arrowwidth=2,
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
                                    figure=network_graph(DEPTH, DEFAULT_CATEGORY))],
            ),

            #########################################right side two output component
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Hover Data**

                            Mouse over values in the graph.
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Click Data**

                            Click on points in the graph.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])


###################################callback for left side components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value, input1):
    # YEAR = value
    # ACCOUNT = input1
    return network_graph(value, input1)
    # to update the global variable of YEAR and ACCOUNT


################################callback for right side components
@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('my-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('my-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=False)
