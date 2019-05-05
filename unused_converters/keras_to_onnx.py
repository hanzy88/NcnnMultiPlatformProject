# -*- coding: utf-8 -*-
import argparse
import os
# from tensorflow.python.keras.models import load_model
from keras.models import load_model
import onnx
import onnxmltools

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--keras_model', help='File path of Input Keras model(*.h5).', required=True)
	parser.add_argument(
			'--onnx_model', help='File path of Output ONNX model(*.onnx).', required=False)
	args = parser.parse_args()
	keras_model_name = args.keras_model
	if args.onnx_model is None:
		onnx_model_name = os.path.splitext(keras_model_name)[0] + '.onnx'
	else:
		onnx_model_name = args.onnx_model

	model = load_model(keras_model_name)
	onnx_model = onnxmltools.convert_keras(model)
	onnx.save(onnx_model, onnx_model_name)

if __name__ == '__main__':
	main()


'''
python keras_to_onnx.py --keras_model mobilenetv2.h5
'''
