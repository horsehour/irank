package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import com.horsehour.math.function.KernelFunction;
import com.horsehour.math.function.LinearKernel;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.PCMModel;
import com.horsehour.util.MathLib;
import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;

/**
 * Pairwise Concordant Margin Classifier
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131024
 */
public class PCM extends Classifier {
	public List<PCMModel> modelList;
	public List<int[]> pairList;
	public Representative rep;

	public List<List<Integer>> cluster;
	public List<Integer> uniq;
	public int nCluster;
	public int dim;

	public KernelFunction kernel;

	/**
	 * 分组代表元类型
	 * 
	 * @author Chunheng Jiang
	 * @version 1.0
	 * @since 20140423
	 */
	public enum Representative {
		MEAN, MAX, MIN, MEDIAN;
	}

	public PCM() {
		super();
		kernel = new LinearKernel();
		rep = Representative.MEAN;
	}

	public PCM(SampleSet trainset, SampleSet testset) {
		this();
		this.trainset = trainset;
		this.valiset = testset;
	}

	public void init() {
		modelList = new ArrayList<>();
		pairList = new ArrayList<>();

		dim = trainset.dim();
		uniq = trainset.getUniqueLabels();
		nCluster = uniq.size();
		cluster = getCluster(trainset, uniq);
	}

	@Override
	public void learn() {
		init();
		List<double[]> repList = new ArrayList<>();
		for (int i = 0; i < nCluster; i++)
			repList.add(getRepresentative(cluster.get(i)));

		for (int i = 0; i < nCluster; i++)
			for (int j = i + 1; j < nCluster; j++) {
				PCMModel model = new PCMModel(repList.get(i), repList.get(j), kernel);
				optimizeModel(cluster.get(i), cluster.get(j), model);
				modelList.add(model);
				pairList.add(new int[] { i, j });
			}

		double precision = eval(trainset);
		System.out.println("Train:" + precision);
	}

	private void optimizeModel(List<Integer> positiveList, List<Integer> negativeList, PCMModel model) {
		List<Double> positivePred = new ArrayList<>();
		List<Double> negativePred = new ArrayList<>();

		for (int pos : positiveList)
			positivePred.add(model.predict(trainset.getSample(pos)));

		for (int neg : negativeList)
			negativePred.add(model.predict(trainset.getSample(neg)));

		Collections.sort(positivePred);
		Collections.sort(negativePred);

		double positiveMax = positivePred.get(positivePred.size() - 1), positiveMin = 0;
		double negativeMax = negativePred.get(negativePred.size() - 1);

		if (positiveMax < negativeMax) {
			model.weight[0] = -1;
			model.weight[1] = 1;

			negativeMax = positiveMax;
			positiveMin = negativePred.get(0);
		} else
			positiveMin = positivePred.get(0);

		model.bias = -(negativeMax + positiveMin) / 2.0;
	}

	/**
	 * 根据样本标签对数据集分组
	 * 
	 * @param sampleset
	 * @param uniqLabel
	 * @return group data set according to their labels
	 */
	private List<List<Integer>> getCluster(SampleSet sampleset, List<Integer> uniqLabel) {
		List<List<Integer>> groupList = new ArrayList<>();
		for (int i = 0; i < uniqLabel.size(); i++)
			groupList.add(new ArrayList<>());

		int size = sampleset.size();
		for (int i = 0; i < size; i++) {
			int idx = uniqLabel.indexOf(sampleset.getLabel(i));
			groupList.get(idx).add(i);
		}
		return groupList;
	}

	/**
	 * 从同一分组中抽取出代表性特征
	 * 
	 * @param indices
	 * @return merge input vectors in the same group
	 */
	private double[] getRepresentative(List<Integer> indices) {
		double[] ret = new double[dim];
		if (rep == Representative.MEAN) {
			for (int i = 0; i < dim; i++)
				ret[i] = MathLib.Data.mean(trainset.getFeatures(i, indices));
		} else if (rep == Representative.MAX) {
			for (int i = 0; i < dim; i++)
				ret[i] = MathLib.Data.max(trainset.getFeatures(i, indices));
		} else if (rep == Representative.MIN) {
			for (int i = 0; i < dim; i++)
				ret[i] = MathLib.Data.min(trainset.getFeatures(i, indices));
		} else if (rep == Representative.MEDIAN) {
			for (int i = 0; i < dim; i++) {
				double[] median = MathLib.Data.median(trainset.getFeatures(i, indices));
				if (median.length == 2)
					ret[i] = (median[0] + median[1]) / 2;
				else
					ret[i] = median[0];
			}
		}
		return ret;
	}

	/**
	 * 使用model预测数据
	 * 
	 * @param sampleset
	 * @return 预测精度
	 */
	@Override
	public double eval(SampleSet sampleset) {
		int count = 0;
		List<Double> outputList;
		for (Sample sample : sampleset.getSamples()) {
			outputList = new ArrayList<>();
			for (PCMModel model : modelList)
				outputList.add(model.predict(sample));
			int pred = majorityVote(outputList);
			if (pred == sample.getLabel())
				count++;
		}
		return (1.0d * count) / sampleset.size();
	}

	/**
	 * @param outputList
	 * @return assign label based on majority vote rule
	 */
	private int majorityVote(List<Double> outputList) {
		int[] pair;
		int[] vote = new int[nCluster];
		for (int i = 0; i < outputList.size(); i++) {
			pair = pairList.get(i);
			if (outputList.get(i) > 0)
				vote[pair[0]]++;
			else
				vote[pair[1]]++;
		}
		int rank = MathLib.getRank(vote, false)[0];
		return uniq.get(rank);
	}

	public static void main(String[] args) throws Exception {
		TickClock.beginTick();

		String data = "/Users/chjiang/Documents/csc/dataset0.txt";

		SampleSet sampleset = Data.loadSampleSet(data);
		int maxTrial = 5;
		List<Callable<Double>> taskList = new ArrayList<>();
		for (int iter = 0; iter < maxTrial; iter++) {
			taskList.add(() -> {
				List<SampleSet> splits = sampleset.splitSamples(0.7F, 0.3F);
				PCM algo = new PCM();
				algo.trainset = splits.get(0);
				algo.learn();
				return algo.eval(splits.get(1));
			});
		}

		List<Double> performances = MulticoreExecutor.run(taskList);
		DoubleSummaryStatistics stats = null;
		stats = performances.stream().collect(Collectors.summarizingDouble(Double::doubleValue));
		System.out.println(stats.toString());

		TickClock.stopTick();
	}
}