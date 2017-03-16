package com.horsehour.ml.classifier.tree.rf;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.horsehour.ml.classifier.Classifier;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * <b>Leo Breiman: Random Forests. Machine Learning 45(1): 5-32 (2001)</b>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20151017
 */
public class RandomForest extends Classifier {
	private ExecutorService exec;
	private final int nThread = Runtime.getRuntime().availableProcessors();

	public int stepSplit = 3;
	public int minSplitNum = 10;
	public int minLeafSize = 5;
	public int nTree = 100;

	private int dim, dimRFS; // dim of random feature subspace
	private List<DecisionTree> model;
	private List<Integer> lblbook;

	public RandomForest() {}

	public void init(){
		lblbook = trainset.getUniqueLabels();
		dim = trainset.dim();
		dimRFS = (int) Math.round(Math.sqrt(dim));// sqrt(m) or log2(m) + 1
		model = new ArrayList<>();
		exec = Executors.newFixedThreadPool(nThread);
	}

	@Override
	public void learn(){
		init();
		List<SampleSet> oobList = new ArrayList<>();
		for (int m = 0; m < nTree; m++) {
			SampleSet[] resample = trainset.bootstrap();// bootstrap and oob
			model.add(new DecisionTree(resample[0], lblbook, stepSplit, minSplitNum, minLeafSize,
			        dimRFS, m));
			oobList.add(resample[1]);
		}

		try {
			exec.invokeAll(model);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		exec.shutdown();

		try {
			exec.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		report(oobList);
	}

	/**
	 * @param oobList
	 */
	public void report(List<SampleSet> oobList){
		StringBuffer sb = new StringBuffer();
		sb.append("id\t tr_accuracy\t oob_accuracy\r\n");

		double[][] ret = new double[nTree][2];
		double[] average = new double[2];
		for (DecisionTree dt : model) {
			int id = dt.getId();
			ret[id][0] = dt.getTrainAccuracy();
			ret[id][1] = dt.eval(oobList.get(id));

			average[0] += ret[id][0];
			average[1] += ret[id][1];
			sb.append(id + "\t" + ret[id][0] + "\t" + ret[id][1] + "\r\n");
		}

		average[0] /= nTree;
		average[1] /= nTree;

		sb.append("average\t" + average[0] + "\t" + average[1] + "\r\n");
		System.out.println(sb.toString());
	}

	@Override
	public double eval(SampleSet sampleset){
		List<Integer> predForest;
		int count = 0;
		for (Sample sample : sampleset.getSamples()) {
			predForest = new ArrayList<>();
			for (DecisionTree tree : model)
				predForest.add(tree.predict(sample));

			int pred = MathLib.Data.mode(predForest).get(0);
			if (pred == sample.getLabel())
				count++;
		}
		return (1.0d * count) / sampleset.size();
	}

	public static void main(String args[]){
		TickClock.beginTick();

		RandomForest rf = new RandomForest();
		rf.stepSplit = 1;
		rf.minSplitNum = 10;
		rf.minLeafSize = 5;

		rf.nTree = 5;

		String data = "Data/classification/iris.dat";
		SampleSet sampleset = Data.loadSampleSet(data);

		List<SampleSet> splitList = null;
		splitList = sampleset.splitSamples(new float[]{1.0F, 1.0F});
		rf.trainset = splitList.get(0);
		rf.learn();
		System.out.println("Pred Accuracy: " + rf.eval(splitList.get(1)));

		TickClock.stopTick();
	}
}
