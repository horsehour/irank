package com.horsehour.ml.data.scale;

import java.util.Collections;
import java.util.List;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;

/**
 * MaxNormalizer
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130101
 */
public class MaxScale extends DataScale {

	@Override
	public void scale(SampleSet sampleSet) {
		List<Sample> samples = sampleSet.getSamples();
		List<Double> features;
		int dim = sampleSet.getSample(0).getDim();

		for (int fid = 0; fid < dim; fid++) {
			features = sampleSet.getFeatureList(fid);
			double maxFeature = Collections.max(features);
			if (maxFeature > 0)
				for (int id = 0; id < samples.size(); id++)
					samples.get(id).setFeature(fid,
					        features.get(id) / maxFeature);
		}
	}
}
