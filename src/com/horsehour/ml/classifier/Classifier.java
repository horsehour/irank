package com.horsehour.ml.classifier;

import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.scale.DataScale;
import com.horsehour.ml.data.scale.IntervalScale;

/**
 * Abstract classifier
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20121204
 */
public abstract class Classifier {
	public SampleSet trainset;
	public SampleSet valiset;

	public DataScale dataScale;

	public int nIter;

	public Classifier() {
		trainset = new SampleSet();
		valiset = new SampleSet();
		dataScale = new IntervalScale(0, 1);
	}

	/**
	 * @param train
	 * @param vali
	 * @param scale
	 */
	public void loadDataSet(SampleSet train, SampleSet vali, boolean scale) {
		trainset = train;
		valiset = vali;

		if (scale) {
			dataScale.scale(trainset);
			dataScale.scale(valiset);
		}
	}

	public void train() {
		double trainPerf = 0, valiPerf = 0, valiStar = 0;
		for (int iter = 0; iter < nIter; iter++) {
			learn();

			trainPerf = eval(trainset);
			valiPerf = eval(valiset);

			if (valiPerf > valiStar)
				valiStar = valiPerf;

			System.out.println(iter + "\t" + trainPerf + "\t" + valiPerf);
		}
		System.out.println("Best performance on vali dataset:" + valiStar);
	}

	public abstract void learn();

	/**
	 * @param sampleset
	 * @return performance on given set
	 */
	public abstract double eval(SampleSet sampleset);

	public String getName() {
		return getClass().getSimpleName();
	}
}
