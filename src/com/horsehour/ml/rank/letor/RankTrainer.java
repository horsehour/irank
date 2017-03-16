package com.horsehour.ml.rank.letor;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.model.Model;
import com.horsehour.util.Messenger;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130311
 */
public abstract class RankTrainer {
	public DataSet trainset;
	public DataSet valiset;

	public Metric trainMetric;
	public Metric valiMetric = new MAP();

	public Model plainModel;// 在习模型
	public Model bestModel;// 习得模型

	public String modelFile;

	public Messenger msg;

	public abstract void init();

	public void train() {
		init();

		int nIter = msg.getNumOfIter();

		double vali = 0;
		double bestvali = Double.NEGATIVE_INFINITY;
		for (int iter = 0; iter < nIter; iter++) {
			learn();
			vali = validate();

			if (vali > bestvali) {
				bestvali = vali;
				updateModel();
			}
		}
		storeModel();
	}

	protected abstract void learn();

	/**
	 * @param dataset
	 * @param metric
	 * @return out-of-sample performance
	 */
	protected double validate(DataSet dataset, Metric metric) {
		double perf = 0;
		Double[] predict;
		int m = dataset.size();
		SampleSet sampleset;

		for (int i = 0; i < m; i++) {
			sampleset = dataset.getSampleSet(i);
			predict = plainModel.predict(sampleset);
			perf += metric.measure(sampleset.getLabels(), predict);
		}

		return perf / m;
	}

	protected double validate() {
		return validate(valiset, valiMetric);
	}

	public abstract void updateModel();

	public abstract void storeModel();

	/**
	 * Load model from local file
	 * 
	 * @param modelFile
	 * @return model saved in model file
	 */
	public abstract Model loadModel(String modelFile);

	/**
	 * @return name
	 */
	public String name() {
		return getClass().getSimpleName();
	}
}
