package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;

import com.horsehour.ml.classifier.tree.rf.DecisionStump;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.TickClock;

/**
 * <p>
 * AdaBoost, short for Adaptive Boosting, is a machine learning algorithm
 * proposed by Freund and Schapire. It is a general method to build a strong
 * classifier out of a set of weak classifiers.
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2012/12/12
 * @see Freund, Y. and R. E. Schapire (1997). "A Decision-Theoretic
 *      Generalization of on-Line Learning and an Application to Boosting "
 *      Journal of Computer and System Sciences 55(1): 119-139.
 */

public class AdaBoost extends Classifier {
	public List<Double> sampleProb;
	public List<DecisionStump> boostModel;

	public List<DecisionStump> weakPool;

	public int bestT = -1;

	public AdaBoost() {
		sampleProb = new ArrayList<Double>();

		boostModel = new ArrayList<DecisionStump>();
		weakPool = new ArrayList<DecisionStump>();
	}

	/**
	 * 初始化
	 */
	public void init() {
		initDistribution();
		buildWeakPool();
	}

	/**
	 * 初始化样本分布
	 */
	private void initDistribution() {
		int sz = trainset.size();
		for (int id = 0; id < sz; id++)
			sampleProb.add(1.0d / sz);
	}

	public void train() {
		double trainPerf = 0, valiPerf = 0, valiStar = 0;
		for (int i = 0; i < nIter; i++) {
			learn();

			trainPerf = eval(trainset);
			valiPerf = eval(valiset);

			if (valiPerf > valiStar) {
				valiStar = valiPerf;
				bestT = boostModel.size() - 1;
			}

			System.out
					.println(i + "\t" + boostModel.get(boostModel.size() - 1).fid + "\t" + trainPerf + "\t" + valiPerf);
		}
		System.out.println("Best performance on vali dataset:" + valiStar);
	}

	public void learn() {
		weakLearn();
		weightWeak();
		updateDistribution();
	}

	/**
	 * 提取特征阈值向量
	 * 
	 * @param vals
	 * @return 特征阈值向量
	 */
	private double[] getFeatureThreshold(List<Double> vals) {
		TreeSet<Double> feature = new TreeSet<>();
		feature.addAll(vals);

		int sz = feature.size();
		double[] threshold = new double[sz - 1];

		double head = feature.pollFirst();
		Iterator<Double> iter = feature.iterator();
		int idx = 0;
		while (iter.hasNext()) {
			threshold[idx] = iter.next();
			idx++;
		}

		threshold[0] = 0.5 * (head + threshold[0]);
		if (sz > 2)
			for (int i = 1; i < sz - 1; i++)
				threshold[i] = 0.5 * (threshold[i - 1] + threshold[i]);

		return threshold;
	}

	/**
	 * 构造弱分类器
	 * 
	 * @return 弱分类器
	 */
	private void weakLearn() {
		DecisionStump weak;

		int id = 0;
		double maxPrecision = 0;
		for (int i = 0; i < weakPool.size(); i++) {
			weak = weakPool.get(i);
			double[] predict = weak.eval(trainset);
			int sz = predict.length;

			double precision = 0;
			for (int j = 0; j < sz; j++)
				if (predict[j] == trainset.getLabel(j))
					precision += sampleProb.get(j);

			if (precision > maxPrecision) {
				maxPrecision = precision;
				id = i;
			}
		}
		weak = weakPool.get(id);
		weak.weight = maxPrecision;
		boostModel.add(weak);
	}

	/**
	 * 给基本模型赋权值
	 */
	private void weightWeak() {
		int id = boostModel.size() - 1;
		DecisionStump weak = boostModel.get(id);
		double precision = weak.weight;

		weak.weight = (double) (0.5 * Math.log(precision / (1 - precision)));
		boostModel.set(id, weak);
	}

	/**
	 * Update distribution by multiplicative weight-update Littlestone-Warmuth
	 * Rule
	 */
	private void updateDistribution() {
		DecisionStump weak = boostModel.get(boostModel.size() - 1);
		double[] predict = weak.eval(trainset);

		double prob = 0, norm = 0;
		int sz = trainset.size();
		for (int i = 0; i < sz; i++) {
			int label = trainset.getSample(i).getLabel();
			prob = (double) Math.exp(-weak.weight * label * predict[i]);
			prob *= sampleProb.get(i);

			sampleProb.set(i, prob);
			norm += prob;
		}

		for (int i = 0; i < sz; i++)
			sampleProb.set(i, sampleProb.get(i) / norm);
	}

	/**
	 * 分裂树
	 * 
	 * @param fid
	 * @param theta
	 * @param subsetL
	 * @param subsetR
	 */
	public void split(int fid, double theta, List<Integer> subsetL, List<Integer> subsetR) {
		int sz = trainset.size();
		subsetL.clear();
		subsetR.clear();

		for (int i = 0; i < sz; i++)
			subsetL.add(i);

		subsetR.addAll(trainset.getSampleIndex(fid, theta));
		sz = subsetR.size();

		for (int i = 0; i < sz; i++)
			subsetL.remove(subsetR.get(i));

	}

	/**
	 * 构建基本模型集合-一个特征对应一个基本模型
	 */
	public void buildWeakPool() {
		int dim = trainset.dim();

		List<Integer> subsetL = new ArrayList<Integer>();
		List<Integer> subsetR = new ArrayList<Integer>();

		double[] thresholds;
		for (int i = 0; i < dim; i++) {
			double maxPrecision = 0, bestTheta = 0;
			int bestLLabel = -1, bestRLabel = 1;

			DecisionStump weak;
			thresholds = getFeatureThreshold(trainset.getFeatureList(i));

			for (double b : thresholds) {
				split(i, b, subsetL, subsetR);
				weak = new DecisionStump(i, b);
				double precision = weak.eval(trainset, subsetL, subsetR);

				if (precision > maxPrecision) {
					bestTheta = b;
					maxPrecision = precision;

					bestLLabel = weak.lLabel;
					bestRLabel = weak.rLabel;
				}
			}

			weak = new DecisionStump(i, bestTheta, maxPrecision);
			weak.lLabel = bestLLabel;
			weak.rLabel = bestRLabel;

			weakPool.add(weak);
		}
	}

	/**
	 * @param sampleset
	 * @return 使用给定数据集评价模型
	 */
	public double eval(SampleSet sampleset) {
		int m = boostModel.size(), hit = 0;
		int sz = sampleset.size();

		double[] predict = new double[sz];

		DecisionStump weak;
		for (int i = 0; i < m; i++) {
			weak = boostModel.get(i);
			predict = weak.eval(sampleset);
			for (int j = 0; j < sz; j++)
				predict[j] += weak.weight * predict[j];
		}

		for (int i = 0; i < sz; i++) {
			if (predict[i] * sampleset.getLabel(i) > 0)
				hit++;
		}
		return (double) hit / sz;
	}

	/**
	 * @param sampleset
	 * @return 预测
	 */
	public double predict(SampleSet sampleset) {
		int hit = 0;
		int sz = sampleset.size();

		double[] predict = new double[sz];

		DecisionStump weak;
		for (int i = 0; i <= bestT; i++) {
			weak = boostModel.get(i);
			predict = weak.eval(sampleset);
			for (int j = 0; j < sz; j++)
				predict[j] += weak.weight * predict[j];
		}

		for (int i = 0; i < sz; i++) {
			if (predict[i] * sampleset.getLabel(i) > 0)
				hit++;
		}
		return (double) hit / sz;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		AdaBoost algo = new AdaBoost();
		algo.nIter = 100;

		String data = "data/classification/biodeg.dat";

		SampleSet sampleset = Data.loadSampleSet(data);

		SampleSet testset = sampleset.pollSamples(0.2f);
		algo.valiset = sampleset.pollSamples(0.2f);
		algo.trainset = sampleset;// 3:1:1

		algo.init();
		algo.train();
		System.out.printf("Best performance is %.2f on the test set.\n", algo.predict(testset));

		TickClock.stopTick();
	}
}