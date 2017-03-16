package com.horsehour.ml.cluster;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 23rd Dec 2015 PM 5:11:59
 * 
 * @see
 *      <li>Rodriguez, A., & Laio, A. (2014).
 *      <a href="http://www.sciencemag.org/content/344/6191/1492.full/">
 *      Clustering by fast search and find of density peaks</a>. Science,
 *      344(6191), 1492-1496. doi:10.1126/science.1242072</li>
 *      <li>An alternative solution provided by <a href=
 *      "http://eric-yuan.me/clustering-fast-search-find-density-peaks/"> Eric
 *      Yuan</a> to select the cut-off distance d_c and calculate the local
 *      density.</li>
 **/
public class DensityPeaks extends Clustering {
	private double[][] distMatrix;

	private double dc;// distance threshold

	private double[] rho;// density
	private int[] orderedIdxRho;// ordered index of density in descend order

	private double[] delta;// min-distance to denser points
	private double[] gamma;// gamma = rho * delta
	private int[] nn;// nearest neighbor
	private BitSet halo;

	// percentage of neighbors on the point set
	public float percent = 0.02F;
	public float deltaRelRatio = 0.35F;
	public int nSample;

	@Override
	public void setup() {
		halo = new BitSet(nSample);
		rho = new double[nSample];
		delta = new double[nSample];
		nn = new int[nSample];

		clusterId = new int[nSample];
		Arrays.fill(clusterId, -1);
	}

	/**
	 * build distance matrix of data points in pair, and fully take advantage of
	 * its symmetric property
	 * 
	 * @param points
	 */
	public void buildDistMatrix(List<double[]> points) {
		nSample = points.size();
		distMatrix = new double[nSample][nSample];
		for (int i = 0; i < nSample; i++) {
			double[] u = points.get(i);
			for (int j = 0; j < nSample; j++) {
				if (j > i) {
					double[] v = points.get(j);
					distMatrix[i][j] = MathLib.Distance.euclidean(u, v);
				} else if (j < i)
					distMatrix[i][j] = distMatrix[j][i];
			}
		}
	}

	/**
	 * Build Distance Matrix from Distance File
	 * 
	 * @param distFile
	 * @param delim
	 */
	public void buildDistMatrix(String distFile) {
		List<double[]> distEntries = Data.loadData(distFile);
		nSample = 0;
		double[] disEntry = distEntries.get(0);
		int minId = (int) disEntry[0], maxId = (int) disEntry[1];
		for (int i = 1; i < distEntries.size(); i++) {
			disEntry = distEntries.get(i);
			if (disEntry[1] > maxId)
				maxId = (int) disEntry[1];
		}
		nSample = maxId - minId + 1;
		distMatrix = new double[nSample][nSample];
		for (int k = 0; k < distEntries.size(); k++) {
			disEntry = distEntries.get(k);
			distMatrix[(int) disEntry[0] - minId][(int) disEntry[1] - minId] = disEntry[2];
		}
	}

	/**
	 * hard cut-off distance threshold d_c as indicated in original paper
	 */
	private void calcDistanceCutoff() {
		double[] dist = new double[nSample * (nSample - 1) / 2];
		int count = 0;

		for (int i = 0; i < nSample - 1; i++)
			for (int j = i + 1; j < nSample; j++) {
				dist[count] = distMatrix[i][j];
				count++;
			}

		int[] rank = MathLib.getRank(dist, true);
		int r = (int) (nSample * (nSample - 1) * percent / 2);
		dc = dist[rank[r]];
	}

	/**
	 * calculate the local density based on Gaussian kernel function
	 */
	public void calcLocalDensityGaussian() {
		for (int i = 0; i < nSample - 1; i++)
			for (int j = i + 1; j < nSample; j++) {
				double temp = distMatrix[i][j] / dc;
				temp = Math.exp(-temp * temp);
				rho[i] += temp;
				rho[j] += temp;
			}
	}

	/**
	 * calculate the local density based on indicator function
	 */
	public void calcLocalDensity() {
		for (int i = 0; i < nSample - 1; i++)
			for (int j = i + 1; j < nSample; j++)
				if (distMatrix[i][j] < dc) {
					rho[i]++;
					rho[j]++;
				}
	}

	/**
	 * calculate the local density based on the proposal from Ryan. He proposed
	 * to compute the local density using mean distance to k nearest neighbors,
	 * and k is selected based on a predefined ratio, which seems to reduce the
	 * sensitivity to the threshold
	 */
	public void calcLocalDensityRyan() {
		int k = (int) (nSample * percent);
		int[] rank;
		for (int i = 0; i < nSample; i++) {
			rank = MathLib.getRank(distMatrix[i], true);
			for (int r = 0; r <= k; r++)
				rho[i] -= distMatrix[i][rank[r]];
			rho[i] /= k;
		}
	}

	/**
	 * calculate the minimum distance from denser points
	 */
	private void calcRelativeDist() {
		orderedIdxRho = MathLib.getRank(rho, false);

		for (int i = 1; i < nSample; i++) { // r[i] is less dense than r[j]
			delta[orderedIdxRho[i]] = Double.MAX_VALUE;
			// j < i
			for (int j = 0; j < i; j++)
				if (distMatrix[orderedIdxRho[i]][orderedIdxRho[j]] < delta[orderedIdxRho[i]]) {
					delta[orderedIdxRho[i]] = distMatrix[orderedIdxRho[i]][orderedIdxRho[j]];
					nn[orderedIdxRho[i]] = orderedIdxRho[j];
				}
			if (delta[orderedIdxRho[i]] > delta[orderedIdxRho[0]])
				delta[orderedIdxRho[0]] = delta[orderedIdxRho[i]];
		}
	}

	/**
	 * calculate gamma
	 */
	private void calcDecisionValue() {
		gamma = new double[nSample];
		for (int i = 0; i < nSample; i++)
			gamma[i] = rho[i] * delta[i];
	}

	@Override
	public void cluster() {
		setup();

		calcDistanceCutoff();

		// calcLocalDensity();
		// calcLocalDensityRyan();
		calcLocalDensityGaussian();

		calcRelativeDist();
		calcDecisionValue();

		if (k > 0)
			detectCenters(k);
		else
			detectCenters();

		buildMembership();
		detectHalo();
	}

	/**
	 * Search centers
	 */
	public void detectCenters() {
		int[] rank = MathLib.getRank(gamma, false);
		// the denst point also has a largest gamma
		double minDelta = delta[rank[0]] * deltaRelRatio;
		for (int i = 0; i < nSample; i++) {
			centers.add(rank[i]);
			clusterId[rank[i]] = k;
			k++;
			if (delta[rank[i]] <= minDelta)
				break;
		}
	}

	public void detectCenters(int k) {
		int[] rank = MathLib.getRank(gamma, false);
		int count = 0;
		for (int i = 0; i < nSample; i++) {
			centers.add(rank[i]);
			clusterId[rank[i]] = count;
			count++;
			if (count >= k)
				break;
		}
	}

	/**
	 * Search centers based on min_rho and min_delta
	 * 
	 * @param minRho
	 * @param minDelta
	 */
	public void detectCenters(double minRho, double minDelta) {
		for (int i = 0; i < nSample; i++)
			if (rho[i] > minRho && delta[i] > minDelta) {
				centers.add(i);
				clusterId[i] = k;
				k++;
			}
	}

	/**
	 * Search centers based on min_gamma
	 * 
	 * @param minGamma
	 */
	public void getCenters(double minGamma) {
		for (int i = 0; i < nSample; i++)
			if (gamma[i] > minGamma) {
				centers.add(i);
				clusterId[i] = k;
				k++;
			}
	}

	/**
	 * Assign each point to a hub based on the class of its denser nearest
	 * neighbor
	 */
	public void buildMembership() {
		for (int i : orderedIdxRho)
			if (clusterId[i] == -1)
				clusterId[i] = clusterId[nn[i]];
	}

	/**
	 * Detect Halo (noise or outlier)
	 */
	public void detectHalo() {
		double[] rhoB = new double[k];// boundary rho
		double meanRho = 0;
		int ci, cj;
		for (int i = 0; i < nSample - 1; i++) {
			ci = clusterId[i];
			for (int j = i + 1; j < nSample; j++) {
				cj = clusterId[j];
				if (ci == cj)
					continue;

				if (distMatrix[i][j] <= dc) {
					meanRho = (rho[i] + rho[j]) / 2;
					if (meanRho > rhoB[ci])
						rhoB[ci] = meanRho;
					if (meanRho > rhoB[cj])
						rhoB[cj] = meanRho;
				}
			}
		}

		for (int i = 0; i < nSample; i++)
			if (rho[i] < rhoB[clusterId[i]])
				halo.set(i);
	}

	/**
	 * summary information about the algorithm
	 * 
	 * @param dest
	 * @throws IOException
	 */
	public void report(String dest) throws IOException {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < nSample; i++) {
			int flag = 0;
			if (halo.get(i))
				flag = 1;
			sb.append(clusterId[i] + "\t" + centers.get(clusterId[i]) + "\t" + flag + "\t" + rho[i] + "\t" + delta[i]
					+ "\t" + gamma[i] + "\r\n");
		}
		FileUtils.write(new File(dest + "/cluster.txt"), sb.toString(), "utf-8", false);
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		DensityPeaks dp = new DensityPeaks();
		String base = "data/research/clustering/";

		List<double[]> data = Data.loadData(base + "aggregation.dat");

		dp.buildDistMatrix(data);
		dp.k = 5;

		dp.cluster();
		dp.report(base);

		TickClock.stopTick();
	}
}