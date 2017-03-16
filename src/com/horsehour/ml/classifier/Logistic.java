package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.LogisticModel;
import com.horsehour.ml.model.Model;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 逻辑回归用于二元分类
 * <p>
 * 正例:+1,负例:0
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131115
 */
public class Logistic extends Classifier {
	private Double[] prob;// 样本属于正例的概率估计值

	public Model model;
	public Model bestModel;

	public float learnRate;// 学习率

	public int sz;
	public int dim;

	public void init(){
		sz = trainset.size();
		prob = new Double[sz];

		dim = trainset.dim();

		model = new LogisticModel(new double[dim], 0);
		bestModel = model.copy();
	}

	/**
	 * 训练模型
	 */
	@Override
	public void train(){
		init();
		double trainPerf = 0, valiPerf = 0, valiStar = 0;
		for (int i = 0; i < nIter; i++) {
			learn();

			trainPerf = eval(trainset);
			valiPerf = eval(valiset);

			if (valiPerf > valiStar) {
				valiStar = valiPerf;
				bestModel = model.copy();
			}
			System.out.println(i + "\t" + trainPerf + "\t" + valiPerf);
		}
	}

	/**
	 * 学习阶段
	 */
	@Override
	public void learn(){
		prob = model.predict(trainset);
		updateWeight();
	}

	/**
	 * 更新参数（批量更新）
	 */
	public void updateWeight(){
		Sample sample;
		double[] deltaWeight = new double[dim];
		double deltaBias = 0;
		double error = 0;
		for (int i = 0; i < sz; i++) {
			sample = trainset.getSample(i);
			error = (sample.getLabel() + 1.0) / 2 - prob[i];
			double[] nabla = MathLib.Matrix.multiply(sample.getFeatures(), error);
			deltaWeight = MathLib.Matrix.add(deltaWeight, nabla);
			deltaBias += error;
		}

		deltaWeight = MathLib.Matrix.multiply(deltaWeight, learnRate);
		((LogisticModel) model).updateWeight(deltaWeight, deltaBias * learnRate);
	}

	/**
	 * @param sampleset
	 * @return 使用给定数据集评价模型
	 */
	@Override
	public double eval(SampleSet sampleset){
		return predict(sampleset, model);
	}

	/**
	 * 最佳模型在测试集上预测
	 * 
	 * @param sampleset
	 * @return 预测精度
	 */
	public double predict(SampleSet sampleset){
		return predict(sampleset, bestModel);
	}

	private double predict(SampleSet sampleset, Model model){
		Double[] predict = model.predict(sampleset);

		int hit = 0;
		int m = sampleset.size();
		for (int i = 0; i < m; i++) {
			int label = sampleset.getLabel(i);
			if ((predict[i] > 0.5 && label == 1) || (predict[i] < 0.5 && label == -1))
				hit++;
		}

		return (1.0d * hit) / m;
	}

	public static void main(String[] args){
		TickClock.beginTick();

		Logistic logit = new Logistic();
		logit.nIter = 10;
		logit.learnRate = 0.001f;

		String data = "course/digits.train";

		SampleSet sampleset = Data.loadSampleSet(data);

		int maxIter = 10;
		List<SampleSet> splitList;
		List<Double> perfList = new ArrayList<>();
		for (int iter = 0; iter < maxIter; iter++) {
			splitList = sampleset.splitSamples(new float[]{0.6F, 0.2F, 0.2F});
			logit.trainset = splitList.get(0);
			logit.valiset = splitList.get(1);
			logit.train();
			perfList.add(logit.predict(splitList.get(2)));
		}

		System.out.println("Avg Test Perf (" + maxIter + ") = " + MathLib.Data.mean(perfList));

		TickClock.stopTick();
	}
}