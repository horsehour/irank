package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.LinearModel;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * Pairwise Concordant Margin Classifier
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131024
 */
public class PC extends OvOMultiClassifier {
	public int dim;

	public PC() {
	}

	@Override
	public void init() {
		super.init();
		nIter = 10;
		dim = trainset.dim();
	}

	/**
	 * Learn optimal binary model using samples of the class pair
	 * 
	 * @param pair
	 * @return optimal model data set of two classes
	 */
	@Override
	public LinearModel learn(Pair<Integer, Integer> pair) {
		int c1 = pair.getKey(), c2 = pair.getValue();
		int size = cluster.get(c1).size() + cluster.get(c2).size();
		double[] probList = new double[size];
		for (int i = 0; i < size; i++)
			probList[i] = 1;

		double[] perfTrace = new double[nIter];
		LinearModel[] modelTrace = new LinearModel[nIter];
		for (int iter = 0; iter < nIter; iter++) {
			LinearModel model = learn(c1, c2, probList);
			double perf = updateProb(c1, c2, model, probList);

			System.out.println("<c" + lblbook.get(c1) + ", c" + lblbook.get(c2) + ">" + " Step" + iter + ":" + perf);

			perfTrace[iter] = perf;
			modelTrace[iter] = model;
		}
		int rank = MathLib.getRank(perfTrace, false)[0];
		return modelTrace[rank];
	}

	/**
	 * @param c1
	 * @param c2
	 * @param probList
	 * @return optimal model based on data of two categories
	 */
	private LinearModel learn(int c1, int c2, double[] probList) {
		double[] weight = getOptimalWeight(c1, c2, probList);
		double bias = getOptimalBias(c1, c2, weight);
		return new LinearModel(weight, bias);
	}

	/**
	 * @param c1
	 * @param c2
	 * @param probList
	 * @return optimal weight of model for class 1 and 2
	 */
	private double[] getOptimalWeight(int c1, int c2, double[] probList) {
		double[] weight = new double[dim];
		for (int i = 0; i < dim; i++) {
			int count = 0;
			for (int id : cluster.get(c1)) {
				weight[i] += trainset.getSample(id).getFeature(i) * probList[count];
				count++;
			}

			for (int id : cluster.get(c2)) {
				weight[i] -= trainset.getSample(id).getFeature(i) * probList[count];
				count++;
			}
		}
		return weight;
	}

	/**
	 * @param c1
	 * @param c2
	 * @param weight
	 * @return optimal bias
	 */
	private double getOptimalBias(int c1, int c2, double[] weight) {
		List<Double> posPred = new ArrayList<Double>();
		List<Double> negPred = new ArrayList<Double>();

		for (int pid : cluster.get(c1))
			posPred.add(MathLib.Matrix.innerProd(trainset.getSample(pid).getFeatures(), weight));

		for (int nid : cluster.get(c2))
			negPred.add(MathLib.Matrix.innerProd(trainset.getSample(nid).getFeatures(), weight));

		Collections.sort(posPred);
		Collections.sort(negPred);

		double posMax = posPred.get(posPred.size() - 1), posMin = posPred.get(0);
		double negMax = negPred.get(negPred.size() - 1), negMin = negPred.get(0);

		// TODO more refined way required
		if (posMax < negMax) {
			MathLib.Matrix.multiply(weight, -1);
			return (negMin + posMax) / 2.0;
		}
		return -(negMax + posMin) / 2.0;
	}

	/**
	 * Update samples' probabilities. If they are correctly classified, their
	 * weights will be reduced, otherwise being increased
	 * 
	 * @param pair
	 * @param model
	 * @param probList
	 * @return accuracy of the model
	 */
	public double updateProb(int c1, int c2, LinearModel model, double[] probList) {
		int correct = 0, count = 0;
		Sample sample;
		for (int pid : cluster.get(c1)) {
			sample = trainset.getSample(pid);
			double pred = model.predict(sample);
			if (pred > 0) {
				correct++;
				probList[count] /= Math.E;// correctly classified, reduce its
											// weight
			} else
				probList[count] *= Math.E;// missclassified, increase its weight
			count++;
		}

		for (int nid : cluster.get(c2)) {
			sample = trainset.getSample(nid);
			double pred = model.predict(sample);
			if (pred < 0) {
				correct++;
				probList[count] /= Math.E;
			} else
				probList[count] *= Math.E;
			count++;
		}

		System.out.println(Arrays.toString(probList));
		int size = cluster.get(c1).size() + cluster.get(c2).size();
		double accuracy = (1.0d * correct) / size;

		if (accuracy < 0.5) {
			model.w = MathLib.Matrix.multiply(model.w, -1);
			model.b *= -1;
			accuracy = 1 - accuracy;
		}
		return accuracy;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		PC algo = new PC();

		String data = "/Users/chjiang/Documents/csc/dataset0.txt";

		SampleSet sampleset = Data.loadSampleSet(data);

		int maxIter = 1;
		List<SampleSet> splitList;
		List<Double> perfList = new ArrayList<>();
		for (int iter = 0; iter < maxIter; iter++) {
			splitList = sampleset.splitSamples(new float[] { 0.7F, 0.3F });
			algo.trainset = splitList.get(0);
			algo.learn();
			perfList.add(algo.eval(splitList.get(1)));
			System.out.println(algo.reportDistribution());
		}

		System.out.println("Avg Test Perf (" + maxIter + ") = " + MathLib.Data.mean(perfList));

		TickClock.stopTick();
	}
}