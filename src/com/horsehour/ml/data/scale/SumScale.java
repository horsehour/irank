package com.horsehour.ml.data.scale;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;

public class SumScale extends DataScale {
	public SumScale() {
	}

	/**
	 * @param sampleSet
	 */
	public void scale(SampleSet sampleSet) {
		int dim = sampleSet.getSample(0).getDim();
		double[] sum = new double[dim];
		for (int fid = 0; fid < dim; fid++) {
			for (int qid = 0; qid < sampleSet.size(); qid++)
				sum[fid] += sampleSet.getSample(qid).getFeature(fid);
		}

		for (int fid = 0; fid < dim; fid++) {
			if (sum[fid] > 0)
				for (Sample sample : sampleSet.getSamples()) {
					double val = sample.getFeature(fid) / sum[fid];
					sample.setFeature(fid, val);
				}
		}
	}
}
