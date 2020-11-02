import os
import numpy as np
import pandas as pd
import igraph

from matplotlib import cm
from matplotlib import colors

def string_tie_graph_to_dataframe(file, output_dir = None):

    # open the file in Read mode
    f = open(file, "r")

    #if f.mode == 'r':
    #    # use f.read() to read file data and store it in variable `contents`
    #    contents = f.read()
    #    print(contents)

    #df_header = ["node_id", "coordinates", "sum_coverage", "length", "coverage", "parent"]
    df_header = ["name", "start", "end", "sum_coverage", "length", "coverage", "parent"]

    # init node set vars
    #node_vars = np.ndarray(shape = (0, 5))
    node_vars = np.ndarray(shape = (0, 6))

    # init edge set vars
    #edge_vars = np.ndarray(shape = (0, 2))
    edge_vars = np.ndarray(shape = (0, 3))
    #edge_set = {}

    # (if your data is too big) read file line by line, readlines() code will
    # segregate your data in easy to read mode.
    f1 = f.readlines()
    for line in f1:
        #print(ll)
        parse_line = line.split(' ')

        # split `coordinate` into `start` and `end` cols and parse rest of columns
        _node_vars = np.concatenate(([parse_line[0]], parse_line[1][1:-2].split('-'), [xx.split('=')[-1] for xx in parse_line[2:5]]))

        # add node set info
        node_vars = np.append(node_vars, np.array([_node_vars]), axis=0)

        # if there `parent_set` is not empty - attention assuming 'trf=' column is empty!
        parent_set = parse_line[6:-1]
        if len(parent_set) > 0:
            # node_id
            child_id = parse_line[0]

            # add `edge` to edge_set
            for parent_id in parent_set:
                # add: ['parent_id', 'child_id', 'weight'=1] for now
                #edge_vars = np.append(edge_vars, np.array([[parent_id, child_id]]), axis=0)
                edge_vars = np.append(edge_vars, np.array([[parent_id, child_id, 1]]), axis=0)

        #import pdb; pdb.set_trace()
        
    # -----------
    # A. Node Set
    # -----------
    
    node_set_df = pd.DataFrame(node_vars, columns=df_header[0:6])
    #node_set_df
    #node_set_df.shape
    
    # rename: ['source', 'tank'] nodes
    node_set_df.at[0, 'name'] = 'source'
    node_set_df.at[len(node_set_df) - 1, 'name'] = 'tank'

    # -----------
    # B. Edge Set
    # -----------
    
    #edge_set_df = pd.DataFrame(edge_vars, columns=['parent', 'child'])
    edge_set_df = pd.DataFrame(edge_vars, columns=['from', 'to', 'weight'])
    #edge_set_df
    #edge_set_df.shape
    
    # store if necessary
    if not output_dir is None:
        node_set_df.to_csv(os.path.join(output_dir, 'splice_graph.nodes'), sep='\t', header=True, index=False)
        edge_set_df.to_csv(os.path.join(output_dir, 'splice_graph.edges'), sep='\t', header=True, index=False)

    return node_set_df, edge_set_df


def layout_splice_graph(g, n_max_tracks = 3, padding = 10):

    n_nodes = g.vcount()

    ## -----------------
    ## A. Track coordinates - Y-axis
    ## -----------------

    # init `node_track` variable - default 'source' and 'tank': 0
    y_node_track = np.zeros(n_nodes)

    # keeps track of last node position in each track of the Y-axis in order to avoid overlap between features
    #y_tracks = [0, 0, 0]
    y_tracks = np.zeros(n_max_tracks)

    # loop over nodes in genomic_region; associate each node with a track that avoids overlaps
    for ii, node in enumerate(g.vs[1:-1]):

        # left bottom x-position
        x_start = node.attributes()['start']
        # right bottom x-position
        x_end = node.attributes()['end']

        track_idx = 0

        # find empty y-track to avoid overlaps
        while (y_tracks[track_idx] != 0) and (y_tracks[track_idx] + padding >= x_start):
            track_idx += 1
        # update current empty track
        y_tracks[track_idx] = x_end

        # associate each node with a track that avoids overlaps
        y_node_track[ii + 1] = track_idx

    ## --------------------
    ## B. Genomic coordinates - X-axis
    ## --------------------

    log_coords = True

    # exclude: 'source', 'tank' nodes
    genomic_coords_start = np.array(g.vs['start'][1:-1])

    if (log_coords):

        # compress big distances between features - with log()
        log_coords_diff_start = np.log(genomic_coords_start[1:] - genomic_coords_start[:-1])

        # add first element
        log_coords_diff_start = np.concatenate([[0], log_coords_diff_start], axis=0)

        # compute coords by adding
        scaled_genomic_coords = np.cumsum(log_coords_diff_start)

    else:
        # do NOT transform coords
        scaled_genomic_coords = genomic_coords_start

    min_coord = min(scaled_genomic_coords)
    max_coord = max(scaled_genomic_coords)
    shift = (2 * max_coord) / n_nodes

    # add: 'source', 'tank' coords
    scaled_genomic_coords =  np.concatenate([[min_coord - shift], scaled_genomic_coords, [max_coord + shift]], axis=0)

    # transform to: [0, 1] interval
    scale_coords = colors.Normalize(vmin=min(scaled_genomic_coords), vmax=max(scaled_genomic_coords))
    scaled_genomic_coords = scale_coords(scaled_genomic_coords)

    layout = list(zip(scaled_genomic_coords, y_node_track))

    return layout


def my_layout_splice_graph(g):

    # get `tree` layout
    layout = g.layout("tree")
    y_coord = max([ss[1] for ss in layout])
    # center tank
    layout[-1][0] = 0

    # exclude: 'source', 'tank' nodes
    genomic_coords = np.array(g.vs['start'][1:-1])

    # compress big gaps
    log_coords_diff = np.log(genomic_coords[1:] - genomic_coords[:-1])
    # add first element
    log_coords_diff = np.concatenate([[0], log_coords_diff], axis=0)

    scaled_genomic_coords = np.cumsum(log_coords_diff)

    max_coord = max(scaled_genomic_coords)
    shift = (2 * max_coord) / len(g.vcount())

    # add: 'source', 'tank' coords
    scaled_genomic_coords =  np.concatenate([[-shift], scaled_genomic_coords, [ max_coord + shift]], axis=0)

    # transform to: [0, 1] interval
    scale_coords = colors.Normalize(vmin=min(scaled_genomic_coords),vmax=max(scaled_genomic_coords))
    scaled_genomic_coords = scale_coords(scaled_genomic_coords) * y_coord

    splice_graph_layout = [[ll[0], scaled_genomic_coords[ii], ] for ii, ll in enumerate(layout)]
    
    return splice_graph_layout


def old_my_layout_splice_graph(g):
    
    # get `tree` layout
    layout = g.layout("tree")
    y_coord = max([ss[1] for ss in layout])
    # center tank
    layout[-1][0] = 0

    # exclude: 'source', 'tank' nodes
    min_coord = min(g.vs['start'][1:-1])
    max_coord = max(g.vs['start'][1:-1])
    length = max_coord - min_coord + 1
    shift = (2 * length) / len(g.vcount())

    scaled_genomic_coords = g.vs['start']
    # modify: 'source', 'tank' nodes
    scaled_genomic_coords[0] = min_coord - shift
    scaled_genomic_coords[-1] = max_coord + shift
    
    # transform to: [0, 1] interval
    scale_coords = colors.Normalize(vmin=min(scaled_genomic_coords),vmax=max(scaled_genomic_coords))
    scaled_genomic_coords = scale_coords(scaled_genomic_coords) * y_coord

    splice_graph_layout = [[ll[0], scaled_genomic_coords[ii], ] for ii, ll in enumerate(layout)]
    
    return splice_graph_layout