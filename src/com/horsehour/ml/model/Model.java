package com.horsehour.ml.model;

import java.io.Serializable;

import org.apache.commons.lang3.SerializationUtils;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;

/**
 * Abstract model including classification, regression, clustering and ranking
 * models
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @since 20130311
 * @since 20131115
 */
public abstract class Model implements Serializable {
	private static final long serialVersionUID = 716029837150978339L;

	/**
	 * Making prediction
	 * 
	 * @param dataset
	 * @return Prediction of input data set
	 */
	public Double[][] predict(DataSet dataset) {
		int sz = dataset.size();
		Double[][] scores = new Double[sz][];
		for (int i = 0; i < sz; i++)
			scores[i] = predict(dataset.getSampleSet(i));
		return scores;
	}

	/**
	 * 预测数据集
	 * 
	 * @param dataset
	 * @return 检索词层级上的标准化预测分值
	 */
	public double[][] normPredict(DataSet dataset) {
		int sz = dataset.size();
		double[][] scores = new double[sz][];
		for (int i = 0; i < sz; i++)
			scores[i] = normPredict(dataset.getSampleSet(i));
		return scores;
	}

	/**
	 * 预测单个检索词关联的多个文档的相关分值
	 * 
	 * @param sampleset
	 * @return 相关分值列表
	 */
	public Double[] predict(SampleSet sampleset) {
		int sz = sampleset.size();
		Double[] score = new Double[sz];

		for (int i = 0; i < sz; i++)
			score[i] = predict(sampleset.getSample(i));

		return score;
	}

	/**
	 * 预测单个检索词关联文档上的相关分值
	 * 
	 * @param sampleset
	 * @return 检索词层级上的标准化预测结果
	 */
	public double[] normPredict(SampleSet sampleset) {
		int sz = sampleset.size();
		double[] score = new double[sz];

		for (int i = 0; i < sz; i++)
			score[i] = predict(sampleset.getSample(i));
		MathLib.Scale.scale(score, 0, 1);
		return score;
	}

	/**
	 * 预测单个样本
	 * 
	 * @param sample
	 * @return 单个样本的预测分值
	 */
	public abstract double predict(Sample sample);

	public Model copy() {
		return (Model) SerializationUtils.clone(this);
	}
}
