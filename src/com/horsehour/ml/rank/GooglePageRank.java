package com.horsehour.ml.rank;

import java.util.Arrays;
import java.util.List;

import com.horsehour.util.Ace;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

public class GooglePageRank {
	int n = 11, nIter = 25;
	float damp = 0.85F;

	float[] scores;
	boolean[][] relation;
	int[] outdegree;

	Ace ace;
	List<String> columnLabels, rowLabels;
	double[][] data;

	public GooglePageRank() {
		relation = new boolean[n][n];
		scores = new float[n];

		for (int i = 0; i < n; i++) {
			scores[i] = 1.0F / n;
			for (int j = 0; j < n; j++)
				relation[i][j] = false;
		}

		relation[1][0] = relation[1][10] = true;
		relation[2][8] = relation[2][10] = true;
		relation[3][8] = relation[3][10] = true;
		relation[4][8] = relation[4][10] = true;
		relation[5][8] = relation[6][8] = relation[7][8] = relation[7][10] = true;
		relation[8][1] = relation[8][7] = relation[8][10] = true;
		relation[9][10] = relation[10][9] = true;

		outdegree = new int[n];
		for (int i = 0; i < n; i++) {
			int count = 0;
			for (int j = 0; j < n; j++) {
				if (j == i)
					continue;
				if (relation[i][j])
					count++;
			}
			outdegree[i] = count;
		}

		rowLabels = Arrays.asList("");
		columnLabels = Arrays.asList("A,B,C,D,E,F,G,H,I,J,K".split(","));
		data = new double[1][n];
	}

	public void compute() {
		float[] pagerank = new float[n];
		Arrays.fill(pagerank, (1.0F - damp) / n);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (j == i)
					continue;
				if (relation[j][i])
					pagerank[i] += damp * scores[j] / outdegree[j];
			}
		}
		MathLib.Scale.sum(pagerank);
		scores = pagerank;
	}

	public void run() {
		for (int i = 0; i < nIter; i++) {
			ace = new Ace("#Iter = " + (i + 1), "/Users/chjiang/Downloads/pr-" + (i - 1) + ".png");
			ace.bar(rowLabels, columnLabels, data);
			for (int k = 0; k < n; k++)
				data[0][k] = scores[k];
			compute();
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		GooglePageRank gpr = new GooglePageRank();
		gpr.run();

		TickClock.stopTick();

	}
}