package com.horsehour.ml.rank.letor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.model.LinearModel;
import com.horsehour.util.MathLib;

/**
 * Pairwise Concordant Ranker
 * 
 * @author Chunheng Jiang
 * @version 4.0
 * @since 20131204
 */
public class PCRank extends DEARank {
	protected void buildCandidatePool() {
		candidatePool = new ArrayList<>();
		String candidateFile = msg.get("candidateFile");

		List<double[]> weightLine = Data.loadData(candidateFile);
		int m = weightLine.size();
		int n = weightLine.get(0).length;

		for (int i = 0; i < m; i++) {
			double[] weight = Arrays.copyOfRange(weightLine.get(i), 1, n);

			if (MathLib.Data.sum(weight) == 0)// 零向量剔除
				continue;

			candidatePool.add(new LinearModel(weight));
		}
	}

	@Override
	public String name() {
		return name() + trainMetric.getName();
	}
}
