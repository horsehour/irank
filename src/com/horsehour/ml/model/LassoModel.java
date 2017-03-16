package com.horsehour.ml.model;

/**
 * A container for arrays and values that are computed during computation of a
 * lasso fit. It also contains the final weights of features.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since May 19, 2014 10:44:59
 **/
public class LassoModel {
	// Number of lambda values
	public int numberOfLambdas;

	// Intercepts
	public double[] intercepts;

	// Compressed weights for each solution
	public double[][] compressedWeights;

	// Pointers to compressed weights
	public int[] indices;

	// Number of weights for each solution
	public int[] numberOfWeights;

	// Number of non-zero weights for each solution
	public int[] nonZeroWeights;

	// The value of lambdas for each solution
	public double[] lambdas;

	// R^2 value for each solution
	public double[] rsquared;

	// Total number of passes over data
	public int numberOfPasses;

	private int numFeatures;

	public LassoModel(int numberOfLambdas, int maxAllowedFeaturesAlongPath, int numFeatures) {
		intercepts = new double[numberOfLambdas];
		compressedWeights = new double[numberOfLambdas][maxAllowedFeaturesAlongPath];
		indices = new int[maxAllowedFeaturesAlongPath];
		numberOfWeights = new int[numberOfLambdas];
		lambdas = new double[numberOfLambdas];
		rsquared = new double[numberOfLambdas];
		nonZeroWeights = new int[numberOfLambdas];
		this.numFeatures = numFeatures;
	}

	public double[] getWeights(int lambdaIdx) {
		double[] weights = new double[numFeatures];
		for (int i = 0; i < numberOfWeights[lambdaIdx]; i++) {
			weights[indices[i]] = compressedWeights[lambdaIdx][i];
		}
		return weights;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		int numberOfSolutions = numberOfLambdas;
		sb.append("Compression R2 values:\n");
		for (int i = 0; i < numberOfSolutions; i++) {
			sb.append((i + 1) + "\t" + nonZeroWeights[i] + "\t" + rsquared[i] + "\t" + lambdas[i] + "\n");
		}
		return sb.toString().trim();
	}
}
