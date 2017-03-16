package com.horsehour.math.opt.gd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.math.function.TransferFunction;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * Gradient Descent Method
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since May 2, 2014 6:30:56 am
 **/
public class GradientDescent {
	int nIter = 1000;
	float eta = 0.001f;
	float epsilon = 1e-20f;

	List<Pair<double[], Double>> searchPath;

	public GradientDescent(int nIter, float step) {
		this.searchPath = new ArrayList<>();
		this.nIter = nIter;
		this.eta = step;
	}

	BiFunction<Double, Double, Double> F = (x1, x2) -> x1 * x1 + 2 * x2 * x2
			+ 2 * Math.sin(2 * Math.PI * x1) * Math.sin(2 * Math.PI * x2);

	BiFunction<Double, Double, double[]> G = (x1, x2) -> {
		return new double[] { 2 * x1 + 4 * Math.PI * Math.cos(2 * Math.PI * x1) * Math.sin(2 * Math.PI * x2),
				4 * x2 + 4 * Math.PI * Math.sin(2 * Math.PI * x1) * Math.cos(2 * Math.PI * x2), };
	};

	public void optimize(double[] x, boolean min) {
		double[] xt = Arrays.copyOf(x, 2);
		double[] g = new double[2];

		int sgn = min ? 1 : -1;

		while (nIter > 0) {
			nIter--;

			g = G.apply(xt[0], xt[1]);
			searchPath.add(Pair.of(xt, F.apply(xt[0], xt[1])));

			xt[0] -= sgn * eta * g[0];
			xt[1] -= sgn * eta * g[1];
		}
	}

	public void optimize(TransferFunction func, double[] input) {
		double[] prev = input;
		double[] next = new double[prev.length];
		while (nIter >= 0) {
			double[] derv = func.calcDerivation(prev);
			next = MathLib.Matrix.lin(prev, 1.0, derv, -eta);
			prev = Arrays.copyOf(next, next.length);
			nIter--;
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		int nIter = 50;
		float[] eta = { 0.01f, 0.1f };
		double[][] x = new double[][] { { 0.1, 0.1 }, { 1, 1 }, { -0.5, -0.5 }, { -1, -1 } };

		for (int i = 0; i < 2; i++) {
			for (int k = 0; k < x.length; k++) {
				GradientDescent gd = new GradientDescent(nIter, eta[i]);
				gd.optimize(x[k], true);
				double[] opt = gd.searchPath.get(nIter - 1).getKey();
				double val = gd.searchPath.get(nIter - 1).getValue();
				System.out.println(Arrays.toString(x[k]) + ", " + Arrays.toString(opt) + ", " + val);
			}
		}

		TickClock.stopTick();
	}
}