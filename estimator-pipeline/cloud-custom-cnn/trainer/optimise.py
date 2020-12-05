import os
import sh

import tensorflow as tf

from tensorflow.python.lib.io import file_io
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python import ops
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph

TRANSFORMS = [
    'remove_nodes(op=Identity)',
    'fold_constants(ignore_errors=true)',
    'merge_duplicate_nodes',
    'strip_unused_nodes',
    'fold_batch_norms'
    ]

##########################################################################################################################################
# Model optimisation functions
##########################################################################################################################################

def get_graph_def_from_saved_model(saved_model_dir):
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session,
            tags=[tag_constants.SERVING],
            export_dir=saved_model_dir
    )
    return meta_graph_def.graph_def

#### Describe GraphDef
def describe_graph(graph_def, show_nodes=False):
    print('Input Feature Nodes: {}'.format([node.name for node in graph_def.node if node.op=='Placeholder']))
    print('')
    print('Unused Nodes: {}'.format([node.name for node in graph_def.node if 'unused'  in node.name]))
    print('')
    print('Output Nodes: {}'.format(
      [node.name for node in graph_def.node if ('predictions' in node.name or 'softmax' in node.name)]))
    print('')
    print('Quantization Nodes: {}'.format( [node.name for node in graph_def.node if 'quant' in node.name]))
    print('')
    print('Constant Count: {}'.format( len([node for node in graph_def.node if node.op=='Const'])))
    print('')
    print('Variable Count: {}'.format( len([node for node in graph_def.node if 'Variable' in node.op])))
    print('')
    print('Identity Count: {}'.format( len([node for node in graph_def.node if node.op=='Identity'])))
    print('')
    print('Total nodes: {}'.format( len(graph_def.node)))
    print('')

    if show_nodes==True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))

#### Get model size and metagraph
def get_size(model_dir, model_file='saved_model.pb', include_vars=True):
    model_file_path = os.path.join(model_dir, model_file)
    print(model_file_path, '')
    pb_size = os.path.getsize(model_file_path)
    variables_size = 0
    if include_vars and os.path.exists(os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
        variables_size = os.path.getsize(os.path.join(
            model_dir,'variables/variables.data-00000-of-00001'))
        variables_size += os.path.getsize(os.path.join(model_dir,'variables/variables.index'))
    print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
    if include_vars:
        print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
        print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))

def get_metagraph(model_dir):
    output = sh.saved_model_cli('show', '--dir=%s' % model_dir, '--all')
    print(output)

#### Get graph def from MetaGraphDef
def get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def optimize_graph(model_dir, graph_filename, transforms, output_node):
    input_names = []
    output_names = [output_node]
    if graph_filename is None:
        graph_def = get_graph_def_from_saved_model(model_dir)
    else:
        graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(
        graph_def,
        input_names,
        output_names,
        transforms)
    tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name='optimized_model.pb')
    print('****************************************')
    print('Graph optimized!')
    print('****************************************')

def freeze_model(saved_model_dir, output_node_names, output_filename):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ''
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags = tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('****************************************')
    print('graph freezed!')
    print('****************************************')

def convert_graph_def_to_saved_model(export_dir, graph_filepath, output_key, output_node_name):
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={
                node.name: session.graph.get_tensor_by_name(
                    '{}:0'.format(node.name))
                for node in graph_def.node if node.op=='Placeholder'},
            outputs={output_key: session.graph.get_tensor_by_name(
                output_node_name)}
        )
    print('****************************************')
    print('Optimized graph converted to SavedModel!')
    print('****************************************')

def optimise_model(EXPORT_DIR):
    
    export_dir = EXPORT_DIR
    print(export_dir)

    saved_model_dir = os.path.join(export_dir, os.listdir(export_dir)[-1])
    print(saved_model_dir)
    describe_graph(get_graph_def_from_saved_model(saved_model_dir))
    get_size(saved_model_dir, 'saved_model.pb')
    get_metagraph(saved_model_dir)

    # freeze model and describe it
    freeze_model(saved_model_dir, 'softmax_tensor', 'frozen_model.pb')
    frozen_filepath = os.path.join(saved_model_dir, 'frozen_model.pb')
    describe_graph(get_graph_def_from_file(frozen_filepath))
    get_size(saved_model_dir, 'frozen_model.pb', include_vars=False)

    # optimize model and describe it
    optimize_graph(saved_model_dir, 'frozen_model.pb', TRANSFORMS, 'softmax_tensor')
    optimized_filepath = os.path.join(saved_model_dir, 'optimized_model.pb')
    describe_graph(get_graph_def_from_file(optimized_filepath))
    get_size(saved_model_dir, 'optimized_model.pb', include_vars=False)

    # convert to saved model and output metagraph again
    optimized_export_dir = os.path.join(export_dir, 'optimized')
    convert_graph_def_to_saved_model(optimized_export_dir, optimized_filepath, 'softmax_tensor',
                                'softmax_tensor:0')
    get_size(optimized_export_dir, 'saved_model.pb')
    get_metagraph(optimized_export_dir)
