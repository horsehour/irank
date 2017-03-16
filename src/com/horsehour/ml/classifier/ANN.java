package com.horsehour.ml.classifier;

import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.nn.Layer;
import com.horsehour.ml.nn.Network;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * Artificial Neural Network
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130510
 */
public class ANN extends Classifier {
	public Network net;
	public Metric trainMetric;
	public List<Integer> labelList;
	public int nInput, nOutput;
	public int nHidden = 10;
	public boolean bias = true;

	private Double[] pred;
	private Layer outputLayer;

	public ANN() {
	}

	@Override
	public void train() {
		labelList = trainset.getUniqueLabels();
		nInput = trainset.dim();
		nOutput = labelList.size();
		pred = new Double[nOutput];
		net = new Network(null, bias, nInput, nOutput, nHidden);
		outputLayer = net.getLayer(net.numLayer - 1);
		super.train();
	}

	@Override
	public void learn() {
		int sz = trainset.size();
		for (int i = 0; i < sz; i++) {
			int[] desireOutput = new int[nOutput];

			Sample sample = trainset.getSample(i);
			net.forwardProp(sample);

			int idx, label = sample.getLabel();
			if ((idx = labelList.indexOf(label)) > -1)
				desireOutput[idx] = 1;

			backwardProp(desireOutput);
		}
	}

	@Override
	public double eval(SampleSet sampleset) {
		int sz = sampleset.size();

		net.clearOutputs();
		int nCorrect = 0;
		for (int i = 0; i < sz; i++) {
			Sample sample = sampleset.getSample(i);
			net.forwardProp(sample);

			for (int idx = 0; idx < nOutput; idx++)
				pred[idx] = outputLayer.getNeuron(idx).getOutput();

			int[] rank = MathLib.getRank(pred, false);
			if (rank[0] == labelList.indexOf(sample.getLabel()))
				nCorrect++;
		}
		return (1.0d * nCorrect) / sz;
	}

	void backwardProp(int[] labels) {
		net.getLayer(net.numLayer - 1).calcOutputLocalGradient(labels);
		int nHiddenLayer = net.numLayer - 2;
		for (int idx = nHiddenLayer; idx > 0; idx--)
			net.getLayer(idx).calcLocalGradient();

		for (Layer layer : net.layers)
			layer.updateWeight();
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		ANN net = new ANN();

		net.nIter = 500;

		String data = "/Users/chjiang/Documents/csc/dataset.txt";
		SampleSet sampleset = Data.loadSampleSet(data);

		SampleSet testset = sampleset.pollSamples(0.3f);
		net.valiset = sampleset.pollSamples(0.3f);
		net.trainset = sampleset;// 3:1:1

		net.train();

		System.out.println("Best performance on test dataset:" + net.eval(testset));

		TickClock.stopTick();
	}
}