import argparse
import tensorflow as tf
import os

# Import python scripts
import trainer.model as model
#import trainer.optimise as optimise

tf.__version__

def main(OUTPUT_DIR):
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    EXPORT_DIR = os.path.join(MODEL_DIR, 'export')

    print(MODEL_DIR)
    print(EXPORT_DIR)
    print(model.TRAIN_DATA)
    print(model.TFRECORDS_TRAIN)

    # Remove previous files
    if tf.gfile.Exists(MODEL_DIR):
        print("Removing previous artifacts...")
        tf.gfile.DeleteRecursively(MODEL_DIR)
    os.makedirs(MODEL_DIR)

    if tf.gfile.Exists(EXPORT_DIR):
        tf.gfile.DeleteRecursively(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)

    
    # Train and evaluate
    estimator = model.run_experiment(MODEL_DIR)
    
    # Export saved_model.pb for serving
    estimator.export_savedmodel(
        export_dir_base=EXPORT_DIR,
        serving_input_receiver_fn=model.serving_input_receiver_fn
    )
    
    #optimise.optimise_model(EXPORT_DIR)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Training file local or GCS')
    parser.add_argument(
        '--eval-file',
        type=str,
        required=True,
        help='Training file local or GCS')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='number of times to go through the data, default=10')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1500,
        help='number of steps per train call, default=200')
    parser.add_argument(
        '--batch-size',
        default=64,
        type=int,
        help='number of records to read during each training step, default=64')
    parser.add_argument(
        '--learning-rate',
        default=0.00003,
        type=float,
        help='learning rate for gradient descent, default=.001')
    parser.add_argument(
        '--num-classes',
        default=8,
        type=int,
        help='number of classes to predict, default=8')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args = parser.parse_args()
    arguments = args.__dict__

    OUTPUT_DIR= arguments.pop('job_dir')
    model.TRAIN_DATA = arguments.pop('train_file')
    model.TEST_DATA = arguments.pop('eval_file')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.NUM_EPOCHS = arguments.pop('num_epochs')
    model.NUM_CLASSES = arguments.pop('num_classes')
    model.LEARNING_RATE = arguments.pop('learning_rate')
    model.NUM_STEPS = arguments.pop('num_steps')
    print ("Will train for {} steps using batch_size={}".format(model.NUM_EPOCHS, model.BATCH_SIZE))
    
    main(OUTPUT_DIR)

  
   

   
    
  
 

    

