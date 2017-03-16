package com.horsehour.ml.rank.letor;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.math.function.PowerFunction;
import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.EnsembleModel;
import com.horsehour.ml.model.FeatureModel;
import com.horsehour.util.MathLib;

/**
 * RankCosine is another Listwise learning to rank approach based on boost
 * framework. The cosine loss function defined is derived from cosine
 * distance/similarity.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2012/12/16
 * @see Qin, T., X.-D. Zhang, et al. (2008). "Query-level loss functions for
 *      information retrieval." Information Processing and Management: an
 *      International Journal 44(2): 838–855.
 */

public class RankCosine extends RankTrainer {

	private Double[][] weight_1, weight_2;
	private Double[][] weakPredict, boostPredict;
	private Double[][] normLabel;

	public RankCosine() {}

	@Override
	public void init(){
		plainModel = new EnsembleModel();
		bestModel = new EnsembleModel();

		normLabel = Data.transLabel(trainset, new PowerFunction(2));
		initWeight();
	}

	private void initWeight(){
		int nTrain = trainset.size();
		weight_1 = new Double[nTrain][];
		weight_2 = new Double[nTrain][];

		for (int qid = 0; qid < nTrain; qid++) {
			int sz = trainset.getSampleSet(qid).size();
			weight_1[qid] = new Double[sz];
			Arrays.fill(weight_1[qid], 1.0 / sz);
			weight_2[qid] = Arrays.copyOf(weight_1[qid], sz);
		}
	}

	@Override
	protected void learn(){
		weakLearn();
		boostPredict();
	}

	/**
	 * 学习Weak Ranker
	 */
	public void weakLearn(){
		int id = 0;
		double preLoss = 0, loss = 0;
		float preWeight = 1.0f, weight = 1.0f;
		int sz = trainset.size();

		weakPredict = new Double[sz][];

		int dim = trainset.dim();
		for (int fid = 0; fid < dim; fid++) {
			SampleSet sampleSet = null;
			for (int qid = 0; qid < sz; qid++) {
				sampleSet = trainset.getSampleSet(qid);
				weakPredict[qid] = sampleSet.getFeatureList(fid).toArray(new Double[sampleSet.size()]);
			}

			if (((EnsembleModel) plainModel).size() > 0) {
				updateWeight();
				weight = weightWeakRanker();
			}

			loss = calcLoss(weight);
			if (fid == 0)
				preLoss = loss;
			if (loss < preLoss) {
				preWeight = weight;
				preLoss = loss;
				id = fid;
			}
		}

		((EnsembleModel) plainModel).addMember(new FeatureModel(id), preWeight);
	}

	/**
	 * 使用Boost Ranker预测训练集
	 */
	public void boostPredict(){
		int m = ((EnsembleModel) plainModel).size();
		double weakWeight = ((EnsembleModel) plainModel).getWeight(m - 1);

		DataSet trainingSet = trainset;
		int nTrain = trainingSet.size();

		if (((EnsembleModel) plainModel).size() == 1) {// The first round
			boostPredict = new Double[nTrain][];
			for (int qid = 0; qid < nTrain; qid++)
				boostPredict[qid] = Arrays.copyOf(weakPredict[qid], weakPredict[qid].length);
			return;
		}

		for (int qid = 0; qid < nTrain; qid++)
			boostPredict[qid] = MathLib.Matrix.lin(boostPredict[qid], 1.0f, weakPredict[qid], weakWeight);
	}

	/**
	 * 根据当前Boost Ranker和Weak Ranker Candidate计算损失
	 * 
	 * @param weakWeight
	 * @return cosineloss
	 */
	public double calcLoss(double weakWeight){
		double loss = 0;
		int sz = trainset.size();
		if (((EnsembleModel) plainModel).size() == 0) {// The first round
			for (int qid = 0; qid < sz; qid++)
				loss += calcCosineLoss(normLabel[qid], weakPredict[qid]);
			return loss;
		}

		for (int qid = 0; qid < sz; qid++) {
			Double[] predict = MathLib.Matrix.lin(boostPredict[qid], 1.0f, weakPredict[qid], weakWeight);
			loss += calcCosineLoss(normLabel[qid], predict);
		}
		return loss;
	}

	/**
	 * 计算余弦损失
	 * 
	 * @param desire
	 * @param predict
	 * @return cosine loss
	 */
	public double calcCosineLoss(Double[] desire, Double[] predict){
		double simCosine = MathLib.Sim.cosine(desire, predict);
		return 0.5 * (1 - simCosine);
	}

	/**
	 * 计算Weak Ranker的权重
	 * 
	 * @return weight of weak ranker
	 */
	public float weightWeakRanker(){
		double a = 0, b = 0, c = 0, d = 0;
		Double[] temp;
		int sz = trainset.size();
		for (int qid = 0; qid < sz; qid++) {
			a += MathLib.Matrix.innerProd(weight_1[qid], weakPredict[qid]);
			c = MathLib.Matrix.innerProd(normLabel[qid], weakPredict[qid]);
			d = MathLib.Matrix.innerProd(weakPredict[qid], weakPredict[qid]);

			temp = MathLib.Matrix.lin(weakPredict[qid], c, normLabel[qid], -1 * d);
			b += MathLib.Matrix.innerProd(weight_1[qid], temp);
		}
		float weight = 0;

		if (b != 0)
			weight = (float) (a / b);
		return weight;
	}

	/**
	 * 根据当前Boost Ranker，更新权重weight_1, weight_2
	 */
	public void updateWeight(){
		double a = 0, b = 0;
		int sz = trainset.size();
		for (int qid = 0; qid < sz; qid++) {
			a = MathLib.Matrix.innerProd(normLabel[qid], boostPredict[qid]);
			b = MathLib.Norm.l2(boostPredict[qid]);

			weight_1[qid] = MathLib.Matrix.lin(boostPredict[qid], a, normLabel[qid], -1 * b);
			weight_1[qid] = MathLib.Matrix.multiply(weight_1[qid], (float) Math.pow(b, 0.75));

			weight_2[qid] = MathLib.Matrix.multiply(boostPredict[qid], (float) Math.pow(b, 0.75));
		}
	}

	@Override
	public void updateModel(){
		bestModel = plainModel.copy();
	}

	@Override
	public void storeModel(){
		try {
			FileUtils.write(new File(modelFile), bestModel.toString(), "",false);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	@Override
	public EnsembleModel loadModel(String modelFile){
		EnsembleModel model = new EnsembleModel();
		List<double[]> lines = Data.loadData(modelFile, "\t");
		int sz = lines.size();
		for (int i = 0; i < sz; i++) {
			double[] line = lines.get(i);
			model.addMember(new FeatureModel((int) line[0]), line[1]);
		}
		return model;
	}

	@Override
	public String name(){
		return name() + "." + trainMetric.getName();
	}
}