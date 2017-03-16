package com.horsehour.ml.data.scale;

import java.util.Collections;
import java.util.List;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;

public class IntervalScale extends DataScale {
	private double a = 0, b = 0;
	public IntervalScale(double a, double b) {
		this.a = a;
		this.b = b;
	}

	/**
	 * @param sampleSet
	 */
	public void scale(SampleSet sampleSet) {
		List<Sample> samples = sampleSet.getSamples();
		List<Double> features;
		int dim = sampleSet.getSample(0).getDim();

		for (int fid = 0; fid < dim; fid++) {
			features = sampleSet.getFeatureList(fid);
			double max = Collections.max(features);
			double min = Collections.min(features);
			if (max > min){
				double alpha = (b - a) / (max - min);
				for (int id = 0; id < samples.size(); id++)
					samples.get(id).setFeature(fid, alpha * (features.get(id) - min) + a);
			}
		}
	}
}