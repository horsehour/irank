package com.horsehour.math.opt.sa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.horsehour.ml.data.Data;
import com.horsehour.util.MathLib;

/**
 * Solving traveling salesman problem based on simulated annealing algorithm
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130714
 */
public class TravelingSalesmanProblem {
	private int[] permutationPrev;
	private int[] permutation;

	private double[][] distances;
	private double shortestDist = 0;

	private int nCities = 0;

	private int nIter = 100;
	private float temperature = 10000F;
	private float minTemperature = 1E-4F;
	// Boltzman constant
	private float cBoltzman = 0.9999F;

	private List<Float> trace;

	public TravelingSalesmanProblem(String cityFile) {
		loadCities(cityFile);
		trace = new ArrayList<>();
	}

	public TravelingSalesmanProblem(int iter, float temp, float minTemp, float bzm, String cityFile) {
		this(cityFile);
		this.nIter = iter;
		this.temperature = temp;
		this.minTemperature = minTemp;
		this.cBoltzman = bzm;
	}

	private void loadCities(String cityFile) {
		List<double[]> datum = Data.loadData(cityFile);
		nCities = datum.size();

		permutationPrev = new int[nCities];
		distances = new double[nCities][nCities];
		for (int i = 0; i < nCities; i++) {
			distances[i] = Arrays.copyOf(datum.get(i), nCities);
			permutationPrev[i] = i;
		}
	}

	/**
	 * @param permu
	 * @return total distance for all nodes in permutation, including a circute
	 *         route
	 */
	private float getDistance(int[] permu) {
		float dist = 0;
		for (int i = 0; i < nCities - 1; i++)
			dist += distances[permu[i]][permu[i + 1]];
		dist += distances[permu[nCities - 1]][0];// 回路距离
		return dist;
	}

	/**
	 * produce new permutation with disturbution
	 * 
	 * @param permutation
	 */
	private void disturbPermut(int[] permutation) {
		permutation = Arrays.copyOf(permutation, nCities);

		int i1 = 0, i2 = 0;
		while (i1 == i2) {
			i1 = (int) MathLib.Rand.uniform(0, nCities);
			i2 = (int) MathLib.Rand.uniform(0, nCities);
		}

		permutation[i1] = permutation[i2];
		permutation[i2] = permutation[i1];
	}

	private void acceptPermution(int[] permu) {
		permutationPrev = Arrays.copyOf(permu, nCities);
	}

	/**
	 * search shortest permutation for salesman
	 */
	public void search() {
		float dist = getDistance(permutationPrev);
		float delta = 0;
		float prob = 0;

		boolean accept = false;

		Random rand = new Random();

		while ((temperature > minTemperature) || (nIter > 0)) {
			disturbPermut(permutationPrev);
			delta = getDistance(permutation) - dist;

			prob = (float) Math.exp(-delta / temperature);
			accept = ((delta < 0) || (delta * (prob - rand.nextFloat()) >= 0));

			if (accept) {
				acceptPermution(permutation);
				dist = delta + dist;
			}

			temperature *= cBoltzman;
			nIter--;

			trace.add(dist);
		}
		shortestDist = dist;
	}

	/**
	 * @return route
	 */
	public String travelRoute() {
		String travelRoute = "";
		for (int i = 0; i < nCities - 1; i++)
			travelRoute += permutationPrev[i] + " -> ";
		travelRoute += permutationPrev[nCities - 1];
		return travelRoute;
	}

	public double getRouteLength() {
		return shortestDist;
	}

	public List<Float> getOptTrace() {
		return trace;
	}
}