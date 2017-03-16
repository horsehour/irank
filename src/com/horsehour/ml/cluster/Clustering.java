package com.horsehour.ml.cluster;

import java.util.ArrayList;
import java.util.List;

/**
 * abstract class for clustering algorithm
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun 10, 2014
 */
public abstract class Clustering {
	public List<double[]> pointList;
	
	public int k = 0;
	public int[] clusterId;
	public List<List<Integer>> clusterList;
	public List<Integer> centers;

	public Clustering() {
		clusterList = new ArrayList<>();
		centers = new ArrayList<>();
	}

	public abstract void setup();

	public abstract void cluster();

	/**
	 * @param groupLabels
	 *            group labels of all data points
	 * @return confusion matrix of data points
	 */
	public int[][] getConfusionMatrix(int[] groupLabels) {
		int nSample = groupLabels.length;
		int[][] matrix = new int[nSample][nSample];
		for (int i = 0; i < nSample; i++) {
			for (int j = 0; j < nSample; j++) {
				if (j > i) {
					if (groupLabels[i] == groupLabels[j])
						matrix[i][j] = 1;
				} else if (j == i)
					matrix[i][j] = 1;
				else
					matrix[i][j] = matrix[j][i];
			}
		}
		return matrix;
	}

	public String getName() {
		return getClass().getSimpleName();
	}
}