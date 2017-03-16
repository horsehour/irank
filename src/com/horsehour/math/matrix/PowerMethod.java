package com.horsehour.math.matrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.ml.metric.KendallTau;
import com.horsehour.util.Ace;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20121207
 */
public class PowerMethod {
	public int nIter = 100;
	public double eps = 1.0E-15;

	public float[] desireProb;
	public float[] predictProb;

	public PowerMethod() {
	}

	/**
	 * @param matrix
	 * @param measuerFile
	 * @param trackFile
	 * @throws IOException
	 */
	public void run(float[][] matrix) {
		float epsilon = 1.0F;
		int len = matrix.length;

		float[] current = new float[len];
		MathLib.Rand.distribution(current);

		float[] previous = Arrays.copyOf(current, len);

		desireProb = computePermutationProb(previous);
		KendallTau metric = new KendallTau();

		double[] x = new double[nIter];
		double[][] y = new double[nIter][3];

		double[][] rankData = new double[nIter][len];

		int count = 0;
		while (count < nIter) {
			for (int i = 0; i < len; i++)
				for (int j = 0; j < len; j++)
					current[i] += matrix[j][i] * previous[j];

			epsilon = computeResidual(current, previous);

			if (epsilon <= eps)
				break;

			predictProb = computePermutationProb(current);

			x[count] = count + 1;
			y[count][0] = epsilon;
			y[count][1] = computeKLDivergence();
			y[count][2] = metric.tauDistance(current, previous);

			previous = Arrays.copyOf(current, current.length);
			desireProb = Arrays.copyOf(predictProb, predictProb.length);

			int[] rank = MathLib.getRank(previous, false);
			for (int k = 0; k < len; k++) {
				rankData[count][k] = rank[k];
			}
			count++;
		}
		new Ace("").combinedLines("Number of Iterations", Arrays.asList("epsilon", "KL-Diverngence", "Kendall Tau"), x,
				y);

		List<String> seriesLabel = new ArrayList<>(len);
		for (int k = 0; k < len; k++)
			seriesLabel.add("r = " + k);

		new Ace("").lines("Number of Iterations", "", seriesLabel, x, rankData);
	}

	/**
	 * @param x
	 * @param y
	 * @return residual of x against y
	 */
	private float computeResidual(float[] x, float[] y) {
		MathLib.Scale.max(x);
		x = MathLib.Matrix.subtract(x, y);
		return MathLib.Norm.l2(x);
	}

	/**
	 * @param ground
	 *            truth
	 * @param vect
	 */
	public float[] computePermutationProb(float[] vect) {
		float expsum = 0;
		int len = vect.length;
		float[] permuteProb = new float[len];

		for (int i = 0; i < len; i++)
			expsum += Math.exp(vect[i]);

		for (int i = 0; i < len; i++)
			permuteProb[i] = (float) (Math.exp(vect[i]) / expsum);

		return permuteProb;
	}

	/**
	 * @return KL-divergence of two distribution
	 */
	public float computeKLDivergence() {
		int len = predictProb.length;
		float loss = 0;
		for (int id = 0; id < len; id++)
			loss += desireProb[id] * Math.log(desireProb[id] / predictProb[id]);
		return loss;
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		PowerMethod pm = new PowerMethod();
		int dim = 200;
		float[][] matrix = new float[dim][dim];
		MathLib.Matrix.stochastic(matrix);

		pm.run(matrix);

		TickClock.stopTick();
	}
}
