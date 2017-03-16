package com.horsehour.ml.model;

import java.util.List;

import com.horsehour.math.function.KernelFunction;
import com.horsehour.math.function.LinearKernel;
import com.horsehour.ml.data.Sample;

/**
 * SVM Model
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130510
 */
public class SVMModel extends Model {
	private static final long serialVersionUID = 7108338451779459780L;

	public KernelFunction kernel = new LinearKernel();
	public List<Double> alphaSV;
	public List<Sample> svs;
	public double bias;
	public int nsv;

	public SVMModel(List<Double> alphaSV, List<Sample> svs, double bias) {
		this.svs = svs;
		this.bias = bias;
		this.nsv = svs.size();
		this.alphaSV = alphaSV;
	}
	
	@Override
	public double predict(Sample sample) {
		Sample sv;
		double score = 0;
		
		for (int i = 0; i < nsv; i++) {
			sv = svs.get(i);
			score += alphaSV.get(i) * sv.getLabel()
			        * kernel.calc(sample.getFeatures(), sv.getFeatures());
		}
		score += bias;
		sample.setScore(score);
		return score;
	}
}
