package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.math.function.IdentityFunction;
import com.horsehour.math.function.LogisticFunction;
import com.horsehour.math.function.TransferFunction;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.nn.Connector;
import com.horsehour.ml.nn.Network;
import com.horsehour.ml.nn.neuron.Neuron;
import com.horsehour.util.MathLib;

/**
 * Extreme Learning Machine
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140425
 */
public class ELM {
	private Network slfn;
	public SampleSet trainset;

	public int nSample;
	public int dim;

	public int task;// 0-regression, 1-classification
	public int nClass;// 类别

	public void init(){
		nSample = trainset.size();
		dim = trainset.dim();

		TransferFunction[] trsFun = {new LogisticFunction(), new IdentityFunction()};
		int[] nHidden = {nClass};

		slfn = new Network(null, false, dim, nHidden, nClass, trsFun);
	}

	public void randomWeight(){
		List<Double> inputWeights = new ArrayList<Double>();
		for (Neuron neuron : slfn.layers.get(0).getNeurons()) {
			List<Connector> outCon = neuron.getOutputConnectors();
			for (int id = 0; id < outCon.size(); id++)
				inputWeights.add(1.0d * MathLib.Rand.uniform(0, 1));
		}
		slfn.netWeights.set(0, inputWeights);
	}
	
	public static void main(String[] args){
		System.out.println();
	}
}
