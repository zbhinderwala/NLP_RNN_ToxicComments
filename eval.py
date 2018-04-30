import tensorflow as tf
import numpy as np
import os
import data_helpers
import ReadData as read
import Config
import csv


# Parameters
# ==================================================

# Data loading params
"""
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

"""
def eval():
    with tf.device('/cpu:0'):
        x_text, y = read.load_data("data/test.csv")

    # Map data into vocabulary
    text_path = os.path.join(Config.checkpoint_dir, "runs","1524954816", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = "/gpfs/home/zbhinderwala/NLP/rnn-text-classification-tf/runs/1524954816/checkpoints/model-100"

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=Config.allow_soft_placement,
            log_device_placement=Config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = read.batch_iteration(list(x_eval), Config.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))

            predictions_human_readable = np.column_stack((np.array(x_text), all_predictions))
            out_path = os.path.join(Config.checkpoint_dir, "..", "prediction.csv")
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
                csv.writer(f).writerows(predictions_human_readable)


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()