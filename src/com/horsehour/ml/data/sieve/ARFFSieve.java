package com.horsehour.ml.data.sieve;

import com.horsehour.ml.data.Sample;

/**
 * 解析weka自定义的arff数据文件(全数值型)
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140418
 */
public class ARFFSieve extends Sieve {
	public ARFFSieve() {}

	public Sample sift(String line) {
		if (line.startsWith("@") || line.startsWith("%"))
			return null;

		String[] segments = line.trim().split(",");

		int dim = segments.length - 1;
		double[] features = new double[dim];
		for (int i = 0; i < dim; i++)
			features[i] = Double.parseDouble(segments[i]);

		int label = (int) Double.parseDouble(segments[dim]);
		
		Sample sample = new Sample(features, label, "");
		sample.setDim(dim);
		return sample;
	}
}
