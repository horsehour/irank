package com.horsehour.ml.metric;

/**
 * <p>
 * 对数似然比（LogLikelihood Ratio,llr）用于度量对象之间的相似关系, 比如统计用户U和V在线观影历史,并构造如下形式的表格:
 * -------------------------------- | V+ V- -------------------------------- U+|
 * k11 k12 U-| k21 k22 --------------------------------
 * 其中,k11表示两个用户皆喜欢的电影数目,k22表示两个都不喜欢的电影数目, k12表示用户U喜欢而用户V不喜欢的电影数目,k21则相反.
 * </p>
 * <p>
 * 对数似然比的计算涉及到信息熵的计算,定义如下: llr(U,V) = 2 * (matrixEntropy - rowEntropy -
 * columnEntropy); 矩阵熵 matrixEntropy = entropy(k11, k12, k21, k22); 行 熵
 * rowEntropy = entropy(k11, k12) + entropy(k21, k22); 列 熵 colEntropy =
 * entropy(k11, k21) + entropy(k12, k22);
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150321
 * @see Mahout
 */
public class LogLikelihoodRatio {
	private double logLikelihoodRatio(int k11, int k12, int k21, int k22) {
		double rowEntropy = entropy(k11, k12) + entropy(k21, k22);
		double columnEntropy = entropy(k11, k21) + entropy(k12, k22);
		double matrixEntropy = entropy(k11, k12, k21, k22);
		return 2 * (matrixEntropy - rowEntropy - columnEntropy);
	}

	private double entropy(int... elements) {
		double sum = 0;
		for (int element : elements)
			sum += element;

		double result = 0.0;
		for (int element : elements) {
			if (element < 0)
				System.err
				        .println("Illegal Negative Count for Entropy Computation.");

			int zeroElement = (element == 0 ? 1 : 0);
			result += element * Math.log((element + zeroElement) / sum);
		}
		return -result;
	}

	public double getLLRSim(int k11, int k12, int k21, int k22) {
		double llr = logLikelihoodRatio(k11, k12, k21, k22);
		return llr / (1 + llr);
	}

	public String name() {
		return "LogLikelihoodRatio";
	}
}
