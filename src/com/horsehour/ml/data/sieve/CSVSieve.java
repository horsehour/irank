package com.horsehour.ml.data.sieve;

import com.horsehour.ml.data.Sample;

/**
 * Comma Seperated Value
 * <p>0.1,0.9,0.1,...,0.3,1</p>
 * The last column is a class label
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131102
 */
public class CSVSieve extends Sieve {

	@Override
	public Sample sift(String line){
		return sift(line, ",");
	}

	public Sample sift(String line, String delim) {
		String[] subs = line.split(delim);
		int dim = subs.length - 1;
		double[] feature = new double[dim];
		for (int i = 0; i < dim; i++)
			feature[i] = Double.parseDouble(subs[i]);
		int label = Integer.parseInt(subs[dim]);
		Sample sample = new Sample(feature, label);
		sample.setDim(dim);
		return sample;
	}
}
