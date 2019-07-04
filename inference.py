
import time
import argparse
import tensorflow as tf

model_filename = "private_detector.frozen.pb"

image_classes = {
    0: "lewd",
    1: "normal",
}

def main(argv):
    FLAGS = argv[0]

    with tf.Graph().as_default() as g:
        def read_image(filename):
            image = tf.read_file(filename)
            return filename, image
        dataset = tf.data.Dataset.from_tensor_slices((FLAGS.filename))
        dataset = dataset.map(read_image, num_parallel_calls=8)
        dataset = dataset.batch(16)

        iterator = dataset.make_initializable_iterator()
        iter_next = iterator.get_next()
        filename_op, image_op = iter_next

        graph_def = tf.GraphDef()

        with tf.gfile.GFile(model_filename, "rb") as f:
            graph_def.ParseFromString(f.read())

        # 'input/images:0' is an input tensor which accepts batch of tf.string, where each element is a binary jpeg/png file
        tf.import_graph_def(graph_def, name='', input_map={'input/images:0': image_op})

        class_index_op = g.get_tensor_by_name('output/class_index:0')
        class_probabilities_op = g.get_tensor_by_name('output/class_probabilities:0')

        with tf.Session(graph=g) as sess:
            sess.run(iterator.initializer)

            files_processed = 0
            start_time = time.time()

            while files_processed < len(FLAGS.filename):
                results = sess.run([filename_op, class_index_op, class_probabilities_op])
                for filename, class_id, probs in zip(*results):
                    filename = str(filename, 'utf-8')
                    print('{}: class: {}, probability: {:.3f}'.format(filename, image_classes[class_id], probs[class_id]))
                    files_processed += 1

            test_time = time.time() - start_time

            print('Total number of images: {}, inference time: {:.1f} seconds, processing speed: {:.1f} images/sec, time per image: {} usec'.format(
                files_processed, test_time, files_processed / test_time, int(test_time * 1000000 / files_processed)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    FLAGS = parser.parse_args()
    tf.app.run(main, argv=(FLAGS,))
