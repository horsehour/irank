package com.horsehour.ml.classifier;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.TickClock;

/**
 * 最大熵的GIS训练算法实现和模型预测
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150315
 */
public class MaxEnt extends Classifier {
	private final List<Instance> instanceList;
	private final List<Pair<String, String>> sampleList;
	private final List<Integer> sampleCount;
	private final List<String> labels;

	private int nSample = 0;
	private int nInstance = 0;
	private int nLabel = 0;

	private float[] weight;
	private float[] prevWeight;
	private float[] empirical;
	private float[] expected;

	public int dim = 0;
	public float epsilon = 1.0E-5F;

	public MaxEnt() {
		instanceList = new ArrayList<>();
		sampleList = new ArrayList<>();
		sampleCount = new ArrayList<>();
		labels = new ArrayList<>();
	}

	/**
	 * @param trainFile
	 */
	public void loadData(String trainFile) {
		List<String> lines = null;
		try {
			lines = FileUtils.readLines(new File(trainFile), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}

		for (String line : lines) {
			if (line.startsWith("#"))
				continue;

			String[] segs = line.split("\t");
			String label = segs[0];
			if (labels.indexOf(label) == -1) {
				labels.add(label);
				nLabel++;
			}

			int len = segs.length;
			if (len - 1 > dim)
				dim = len - 1;

			List<String> feature = new ArrayList<String>();
			for (int i = 1; i < len; i++) {
				feature.add(segs[i]);
				Pair<String, String> pair = Pair.of(label, segs[i]);
				int idx = sampleList.indexOf(pair);
				if (idx == -1) {
					sampleList.add(pair);
					sampleCount.add(1);// append
					nSample++;
				} else
					sampleCount.set(idx, sampleCount.get(idx) + 1);// increase
			}

			Instance instance = new Instance(label, feature);
			instanceList.add(instance);
			nInstance++;
		}
	}

	public Instance[] loadInstances(String dataFile) {
		List<String> lines = null;
		try {
			lines = FileUtils.readLines(new File(dataFile), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		int size = 0;
		for (String line : lines) {
			if (line.startsWith("#"))
				continue;
			size++;
		}

		Instance[] instances = new Instance[size];
		int count = 0;
		for (String line : lines) {
			if (line.startsWith("#"))
				continue;

			String[] splits = line.split("\t");
			String label = splits[0];

			List<String> feature = new ArrayList<String>();
			for (int i = 1; i < splits.length; i++)
				feature.add(splits[i]);

			Instance instance = new Instance(label, feature);
			instances[count] = instance;
			count++;
		}
		return instances;
	}

	/**
	 * 初始化： 计算特征函数的经验期望 - 特征函数出现的次数/样本数
	 */
	public void init() {
		nIter = 100;

		weight = new float[nSample];
		prevWeight = new float[nSample];

		empirical = new float[nSample];
		for (int i = 0; i < nSample; i++)
			empirical[i] = (1.0F * sampleCount.get(i)) / nInstance;
	}

	@Override
	public void train() {
		init();
		int step = 0;
		do {
			System.out.print("step " + (++step) + ": ");
			getExpect();
			for (int k = 0; k < nSample; k++) {
				prevWeight[k] = weight[k];
				weight[k] += 1.0F / dim * Math.log(empirical[k] / expected[k]);
			}
			System.out.println(Arrays.toString(weight));
		} while (--nIter > 0 && !converge(prevWeight, weight));
	}

	/**
	 * @param w1
	 * @param w2
	 * @return true if converge
	 */
	private boolean converge(float[] w1, float[] w2) {
		float dist = 0;
		for (int i = 0; i < w1.length; i++)
			dist += (w1[i] - w2[i]) * (w1[i] - w2[i]);
		if (Math.sqrt(dist) <= epsilon)
			return true;
		return false;
	}

	/**
	 * predictor' expected value
	 */
	public void getExpect() {
		expected = new float[nSample];

		Pair<String, String> pair = null;
		for (int i = 0; i < nInstance; i++) {
			List<String> features = instanceList.get(i).feature;
			float[] prob = predict(features);
			for (int k = 0; k < nLabel; k++) {
				for (String feature : features) {
					pair = Pair.of(labels.get(k), feature);
					int idx = sampleList.indexOf(pair);
					if (idx == -1)
						continue;

					expected[idx] += prob[k] * (1.0 / nInstance);
				}
			}
		}
	}

	/**
	 * @param features
	 * @return p(y|x)
	 */
	public float[] predict(List<String> features) {
		float[] prob = new float[nLabel];
		float sum = 0;
		Pair<String, String> pair = null;
		for (int i = 0; i < nLabel; i++) {
			float weightSum = 0;
			for (String feature : features) {
				pair = Pair.of(labels.get(i), feature);
				int idx = sampleList.indexOf(pair);
				if (idx == -1)
					continue;

				weightSum += weight[idx];
			}
			prob[i] = (float) Math.exp(weightSum);
			sum += prob[i];
		}

		for (int i = 0; i < nLabel; i++)
			prob[i] /= sum;

		return prob;
	}

	public List<float[]> predict(Instance[] instances) {
		List<float[]> ret = new ArrayList<float[]>();
		List<String> features = null;
		for (int m = 0; m < instances.length; m++) {
			features = instances[m].feature;

			float[] prob = new float[nLabel];
			float sum = 0;
			Pair<String, String> pair = null;
			for (int i = 0; i < nLabel; i++) {
				float weightSum = 0;
				for (String feature : features) {
					pair = Pair.of(labels.get(i), feature);
					int idx = sampleList.indexOf(pair);
					if (idx == -1)
						continue;

					weightSum += weight[idx];
				}
				prob[i] = (float) Math.exp(weightSum);
				sum += prob[i];
			}

			for (int i = 0; i < nLabel; i++)
				prob[i] /= sum;
			ret.add(prob);
		}
		return ret;
	}

	public class Instance {
		public String label;
		public List<String> feature = new ArrayList<String>();

		public Instance(String label, List<String> fieldList) {
			this.label = label;
			this.feature = fieldList;
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String trainFile = "data/research/classification/maxent-train.dat";
		MaxEnt maxent = new MaxEnt();
		maxent.loadData(trainFile);
		maxent.train();

		List<String> fieldList = new ArrayList<String>();
		fieldList.add("2:1");
		fieldList.add("1:2");

		float[] prob = maxent.predict(fieldList);
		System.out.println(Arrays.toString(prob));

		TickClock.stopTick();
	}

	@Override
	public void learn() {
	}

	@Override
	public double eval(SampleSet sampleset) {
		return 0;
	}
}
