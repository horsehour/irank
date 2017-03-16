package com.horsehour.ml.nn;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.math.function.TransferFunction;
import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.Model;
import com.horsehour.ml.nn.neuron.Neuron;

/**
 * @author Chunheng Jiang
 * @version 3.0
 * @see Simon Haykin, Neural Networks and Learning Machines, 3rd edition,
 *      p122-38
 * @since 20131216
 */
public class Network extends Model {
	private static final long serialVersionUID = 3033820560795032478L;

	public List<Layer> layers;
	public List<List<Double>> netWeights;
	public int numLayer = 2;

	protected Algo user;
	protected boolean bias = false;
	protected int numOutputNeuron = 1;

	public Network(Network net) {
		layers = net.layers;
		numLayer = net.numLayer;
		numOutputNeuron = net.numOutputNeuron;

		bias = net.bias;
		user = net.user;
		netWeights = net.netWeights;
	}

	/**
	 * @param user
	 * @param nInput
	 * @param nHidden
	 * @param nOutput
	 * @param bias
	 */
	public Network(Algo user, boolean bias, int nInput, int nOutput, int... nHidden) {
		this.user = user;
		this.bias = bias;

		this.numOutputNeuron = nOutput;
		layers = new ArrayList<>();

		if (bias)
			nInput += 1;// plus bias

		layers.add(new Layer(nInput, user));// input layer

		int nHiddenLayer;
		if (nHidden == null)
			nHiddenLayer = 0;
		else
			nHiddenLayer = nHidden.length;

		numLayer = nHiddenLayer + 2;

		for (int i = 0; i < nHiddenLayer; i++)
			layers.add(new Layer(nHidden[i], user));// hidden layer

		layers.add(new Layer(nOutput, user));// output layer

		connectNet();
	}

	public Network(Algo user, boolean bias, int nInput, int[] nHidden, int nOutput, TransferFunction[] functions) {
		this.user = user;
		this.bias = bias;

		this.numOutputNeuron = nOutput;
		layers = new ArrayList<>();

		if (bias)
			nInput += 1;
		layers.add(new Layer(nInput, user));

		int nHiddenLayer;
		if (nHidden == null)
			nHiddenLayer = 0;
		else
			nHiddenLayer = nHidden.length;

		numLayer = nHiddenLayer + 2;

		for (int i = 0; i < nHiddenLayer; i++)
			layers.add(new Layer(nHidden[i], user, functions[i]));
		layers.add(new Layer(nOutput, user, functions[nHiddenLayer]));
		connectNet();
	}

	void connectNet() {
		Layer inputLayer = layers.get(0);
		inputLayer.connectTo(layers.get(1));

		if (bias) {
			Neuron biasNeuron = inputLayer.getNeuron(inputLayer.size() - 1);
			for (int id = 1; id < numLayer - 1; id++)
				biasNeuron.connectTo(layers.get(id + 1));
		}

		for (int id = 1; id < numLayer - 1; id++) {
			Layer layer = layers.get(id);
			layer.connectTo(layers.get(id + 1));
		}
	}

	public void forwardProp(Sample sample) {
		Layer inputLayer = layers.get(0);
		int numBias = bias ? 1 : 0;

		List<Neuron> neurons = inputLayer.getNeurons();
		for (int idx = 0; idx < neurons.size() - numBias; idx++)
			neurons.get(idx).addOutput(sample.getFeature(idx));

		if (bias)
			neurons.get(neurons.size() - 1).addOutput(1.0f);// for bias

		for (int idx = 1; idx < numLayer; idx++)
			layers.get(idx).propagate();
	}

	public void clearOutputs() {
		for (Layer layer : layers)
			for (Neuron neuron : layer.getNeurons())
				neuron.clearOutputs();
	}

	public int getNumOutputNeuron() {
		return numOutputNeuron;
	}

	public Layer getLayer(int layerId) {
		return layers.get(layerId);
	}

	public void setLearningRate(float lr) {
		for (Layer layer : layers)
			for (Neuron neuron : layer.getNeurons())
				neuron.setLearningRate(lr);
	}

	public float getLearningRate() {
		return layers.get(0).getNeuron(0).getLearningRate();
	}

	public void updateWeight() {
		netWeights = new ArrayList<>();
		for (int idx = 0; idx < numLayer - 1; idx++) {
			List<Double> weights = new ArrayList<>();

			for (Neuron neuron : layers.get(idx).getNeurons()) {
				List<Connector> outCon = neuron.getOutputConnectors();
				for (int id = 0; id < outCon.size(); id++)
					weights.add(outCon.get(id).getWeight());
			}
			netWeights.add(weights);
		}
	}

	public void reweight() {
		for (int idx = 0; idx < numLayer - 1; idx++) {
			List<Double> weights = netWeights.get(idx);
			int count = 0;
			for (Neuron neuron : layers.get(idx).getNeurons()) {
				for (int id = 0, sz = neuron.getOutputConnectors().size(); id < sz; id++) {
					neuron.getOutputConnectors().get(id).setWeight(weights.get(count));
					count++;
				}
			}
		}
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < numLayer - 1; i++) {
			List<Double> weights = netWeights.get(i);
			for (double weight : weights)
				sb.append(weight + "\t");
			sb.append("\r\n");
		}
		return sb.toString();
	}

	public Double[][] predict(DataSet dataset) {
		reweight();
		return super.predict(dataset);
	}

	public Double[] predict(SampleSet sampleset) {
		clearOutputs();
		Sample sample;
		for (int id = 0; id < sampleset.size(); id++) {
			sample = sampleset.getSample(id);
			forwardProp(sample);
		}
		List<Double> pred;
		Layer outputLayer;
		Neuron outputNeuron;

		outputLayer = layers.get(layers.size() - 1);
		outputNeuron = outputLayer.getNeuron(0);

		pred = outputNeuron.getOutputList();

		Double[] ret = new Double[pred.size()];
		for (int i = 0; i < pred.size(); i++)
			ret[i] = pred.get(i);
		return ret;
	}

	@Override
	public double predict(Sample sample) {
		return 0;
	}
}