package com.horsehour.ml.model;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.metric.Metric;

/**
 * 使用验证集选择训练模型
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140102
 */
public class ModelSelection {
	public DataSet valiset;// validation set
	public Metric metric;// validation metric

	public double validate(Model model) {
		Double[] predict;
		double score = 0;
		int m = valiset.size();
		SampleSet sampleset;
		for (int i = 0; i < m; i++) {
			sampleset = valiset.getSampleSet(i);
			predict = model.predict(sampleset);
			score += metric.measure(sampleset.getLabels(), predict);
		}
		return score / m;
	}
}
