package com.horsehour.ml.dl;

import java.util.Random;

public class SDA {
	public int nSample;
	public int nInput;
	public int[] nHiddenLayers;
	public int nOutput;
	public int nLayer;
	public HiddenLayerDiscrete[] layersSigmoid;
	public DA[] layersDA;
	public LogisticRegressionDiscrete layerLogistic;
	public Random rnd;

	public SDA(int n, int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, Random rnd) {
		int input_size;

		this.nSample = n;
		this.nInput = n_ins;
		this.nHiddenLayers = hidden_layer_sizes;
		this.nOutput = n_outs;
		this.nLayer = n_layers;

		this.layersSigmoid = new HiddenLayerDiscrete[n_layers];
		this.layersDA = new DA[n_layers];

		if (rnd == null)
			this.rnd = new Random(1234);
		else
			this.rnd = rnd;

		// construct multi-layer
		for (int i = 0; i < this.nLayer; i++) {
			if (i == 0) {
				input_size = this.nInput;
			} else {
				input_size = this.nHiddenLayers[i - 1];
			}

			// construct sigmoid_layer
			this.layersSigmoid[i] = new HiddenLayerDiscrete(this.nSample, input_size, this.nHiddenLayers[i], null, null,
					rnd);

			// construct dA_layer
			this.layersDA[i] = new DA(this.nSample, input_size, this.nHiddenLayers[i], this.layersSigmoid[i].W,
					this.layersSigmoid[i].b, null, rnd);
		}

		// layer for output using Logistic Regression
		this.layerLogistic = new LogisticRegressionDiscrete(this.nSample, this.nHiddenLayers[this.nLayer - 1],
				this.nOutput);
	}

	public void pretrain(int[][] train_X, double lr, double corruption_level, int epochs) {
		int[] layer_input = new int[0];
		int prev_layer_input_size;
		int[] prev_layer_input;

		for (int i = 0; i < nLayer; i++) { // layer-wise
			for (int epoch = 0; epoch < epochs; epoch++) { // training epochs
				for (int n = 0; n < nSample; n++) { // input x1...xN
					// layer input
					for (int l = 0; l <= i; l++) {

						if (l == 0) {
							layer_input = new int[nInput];
							for (int j = 0; j < nInput; j++)
								layer_input[j] = train_X[n][j];
						} else {
							if (l == 1)
								prev_layer_input_size = nInput;
							else
								prev_layer_input_size = nHiddenLayers[l - 2];

							prev_layer_input = new int[prev_layer_input_size];
							for (int j = 0; j < prev_layer_input_size; j++)
								prev_layer_input[j] = layer_input[j];

							layer_input = new int[nHiddenLayers[l - 1]];

							layersSigmoid[l - 1].sample_h_given_v(prev_layer_input, layer_input);
						}
					}

					layersDA[i].train(layer_input, lr, corruption_level);
				}
			}
		}
	}

	public void finetune(int[][] train_X, int[][] train_Y, double lr, int epochs) {
		int[] layer_input = new int[0];
		// int prev_layer_input_size;
		int[] prev_layer_input = new int[0];

		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int n = 0; n < nSample; n++) {

				// layer input
				for (int i = 0; i < nLayer; i++) {
					if (i == 0) {
						prev_layer_input = new int[nInput];
						for (int j = 0; j < nInput; j++)
							prev_layer_input[j] = train_X[n][j];
					} else {
						prev_layer_input = new int[nHiddenLayers[i - 1]];
						for (int j = 0; j < nHiddenLayers[i - 1]; j++)
							prev_layer_input[j] = layer_input[j];
					}

					layer_input = new int[nHiddenLayers[i]];
					layersSigmoid[i].sample_h_given_v(prev_layer_input, layer_input);
				}

				layerLogistic.train(layer_input, train_Y[n], lr);
			}
			// lr *= 0.95;
		}
	}

	public void predict(int[] x, double[] y) {
		double[] layer_input = new double[0];
		// int prev_layer_input_size;
		double[] prev_layer_input = new double[nInput];
		for (int j = 0; j < nInput; j++)
			prev_layer_input[j] = x[j];

		double linear_output;

		// layer activation
		for (int i = 0; i < nLayer; i++) {
			layer_input = new double[layersSigmoid[i].n_out];

			for (int k = 0; k < layersSigmoid[i].n_out; k++) {
				linear_output = 0.0;

				for (int j = 0; j < layersSigmoid[i].n_in; j++) {
					linear_output += layersSigmoid[i].W[k][j] * prev_layer_input[j];
				}
				linear_output += layersSigmoid[i].b[k];
				layer_input[k] = Utils.sigmoid(linear_output);
			}

			if (i < nLayer - 1) {
				prev_layer_input = new double[layersSigmoid[i].n_out];
				for (int j = 0; j < layersSigmoid[i].n_out; j++)
					prev_layer_input[j] = layer_input[j];
			}
		}

		for (int i = 0; i < layerLogistic.n_out; i++) {
			y[i] = 0;
			for (int j = 0; j < layerLogistic.n_in; j++) {
				y[i] += layerLogistic.W[i][j] * layer_input[j];
			}
			y[i] += layerLogistic.b[i];
		}

		layerLogistic.softmax(y);
	}

	private static void test_sda() {
		Random rng = new Random(123);

		double pretrain_lr = 0.1;
		double corruption_level = 0.3;
		int pretraining_epochs = 1000;
		double finetune_lr = 0.1;
		int finetune_epochs = 500;

		int train_N = 10;
		int test_N = 4;
		int n_ins = 28;
		int n_outs = 2;
		int[] hidden_layer_sizes = { 15, 15 };
		int n_layers = hidden_layer_sizes.length;

		// training data
		int[][] train_X = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1 } };

		int[][] train_Y = { { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 },
				{ 0, 1 } };

		// construct SdA
		SDA sda = new SDA(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, rng);

		// pretrain
		sda.pretrain(train_X, pretrain_lr, corruption_level, pretraining_epochs);

		// finetune
		sda.finetune(train_X, train_Y, finetune_lr, finetune_epochs);

		// test data
		int[][] test_X = { { 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 } };

		double[][] test_Y = new double[test_N][n_outs];

		// test
		for (int i = 0; i < test_N; i++) {
			sda.predict(test_X[i], test_Y[i]);
			for (int j = 0; j < n_outs; j++) {
				System.out.print(test_Y[i][j] + " ");
			}
			System.out.println();
		}
	}

	public static void main(String[] args) {
		test_sda();
	}
}
