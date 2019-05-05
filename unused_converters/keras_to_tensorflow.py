# -*- coding: utf-8 -*-
import argparse
import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--keras_model', help='File path of Input Keras model(*.h5).', required=True)
	parser.add_argument(
			'--export_dir', help='Directory path of Output PB and variables(*.pb, variables).', required=False)
	args = parser.parse_args()
	keras_model_name = args.keras_model
	if args.export_dir is None:
		export_dir_name = 'export_' + os.path.splitext(keras_model_name)[0]
	else:
		export_dir_name = args.export_dir

	old_session = tf.keras.backend.get_session()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	tf.keras.backend.set_session(sess)
	model = load_model(keras_model_name)
	builder = tf.saved_model.builder.SavedModelBuilder(export_dir_name)
	signature = tf.saved_model.predict_signature_def(inputs={t.name:t for t in model.inputs},
												  outputs={t.name:t for t in model.outputs})
	builder.add_meta_graph_and_variables(sess,
									  tags=[tf.saved_model.tag_constants.SERVING],
									  signature_def_map={'predict': signature})
	builder.save(as_text=True)
	sess.close()
	tf.keras.backend.set_session(old_session)
	
	inputs = [t.name for t in model.inputs]
	outputs = [t.name for t in model.outputs]
	print('input_node_names: ' + ', '.join(inputs))
	print('output_node_names: ' + ','.join(outputs))

	outputs = [t.name.split(':')[0] for t in model.outputs]
	freeze_command = 'freeze_graph ' \
		 + ' --input_saved_model_dir=' + export_dir_name \
		 + ' --output_graph=' + os.path.splitext(keras_model_name)[0] + '.pb' \
		 + ' --output_node_names=' + ', '.join(outputs) \
		 + ' --clear_devices'
	print('---enter the following command to freeze the graph---')
	print(freeze_command)

if __name__ == '__main__':
	main()

'''
python keras_to_tensorflow.py --keras_model vgg16.h5

freeze_graph ^
--input_saved_model_dir=./conv_mnist_pb ^
--output_graph=vgg16.pb ^
--output_node_names=predictions/Softmax ^
--clear_devices

python -m tf2onnx.convert ^
    --input vgg16.pb ^
    --saved-model vgg16.onnx
'''

