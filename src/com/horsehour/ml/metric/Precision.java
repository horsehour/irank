package com.horsehour.ml.metric;

import java.util.List;

import com.horsehour.util.MathLib;

/**
 * Information retrieval associates to search relevant documents. However, the
 * retrieval tool used makes some mistakes. It returns a lot of documents after
 * the user submits her query. Some of the returned documents may be irrelevant
 * to the query, some of them may be relevant. We expect that the retrieval
 * system could be able to fetch as many relevant documents as possible. The
 * metric used to measure such performance is precision. It is the percentage of
 * relevant and correctly selected documents to all returned documents.
 * 
 * @author Chunheng Jiang
 * @version 4.0
 * @since 20131204
 * @see <a href="http://research.microsoft.com/en-us/um/beijing/
 *      projects/letor/LETOR4.0/Evaluation/Eval-Score-4.0.pl.txt">LETOR4.0
 *      Eval</a>
 */
public class Precision extends Metric {
	private int k = 10;
	// r = 0, rel = 0; r >= 1, rel = 1
	private int[] rel = { 0, 1, 1, 1 };

	public Precision() {}

	public Precision(int k) {
		this.k = k;
	}

	@Override
	public double measure(List<? extends Number> desire, List<? extends Number> predict) {
		Double[] topP = getTopKPrecision(desire, predict);
		return topP[k - 1];
	}

	public Double[] getTopKPrecision(List<? extends Number> desire, List<? extends Number> predict) {
		return getTopKPrecision(MathLib.linkedSort(desire, predict, false));
	}

	private Double[] getTopKPrecision(List<? extends Number> label) {
		Double[] precisionAtN = new Double[k];
		int sz = label.size();
		int nRel = 0;
		for (int i = 0; i < k; i++) {
			int r = 0;
			if (i < sz)
				r = label.get(i).intValue();
			if (rel[r] == 1)
				nRel++;
			precisionAtN[i] = nRel * 1.0 / (i + 1);
		}
		return precisionAtN;
	}

	@Override
	public String getName() {
		return super.getName() + "@" + k;
	}
}
