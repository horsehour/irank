package com.horsehour.ml.model;

import com.horsehour.math.function.KernelFunction;
import com.horsehour.math.function.LinearKernel;
import com.horsehour.ml.data.Sample;

/**
 * Model for PCM classifier
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140422
 */
public class PCMModel extends Model {
	private static final long serialVersionUID = -5169936703519814374L;

	public double[] positiveRep, negativeRep;
	public double[] weight = { 1.0, -1.0 };
	public KernelFunction kernel = new LinearKernel();
	public double bias = 0;

	public PCMModel(double[] positiveRep, double[] negativeRep) {
		this.positiveRep = positiveRep;
		this.negativeRep = negativeRep;
	}

	public PCMModel(double[] positiveRep, double[] negativeRep,
	        KernelFunction kernel) {
		this(positiveRep, negativeRep);
		this.kernel = kernel;
	}

	@Override
	public double predict(Sample sample) {
		double pred = weight[0] * kernel.calc(sample.getFeatures(), positiveRep)
        + weight[1] * kernel.calc(sample.getFeatures(), negativeRep)
        + bias;
		sample.setScore(pred);
		return pred;
	}
}
