package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.RateSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 专司统计之责
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 9:05:27 PM Mar 25, 2015
 */
public class Statistician {
	public static String pivotFile = "data/research/movielens-100k-pivot/";

	/**
	 * 汇报统计元数据(用户数、项目数、评分数等)
	 * 
	 * @param rateSet
	 * @throws IOException 
	 */
	public static void reportMetaData(RateSet rateSet) throws IOException{
		StringBuffer sb = new StringBuffer();
		int nUser = rateSet.nUser;
		int nItem = rateSet.nItem;
		int nRate = rateSet.nRate;

		sb.append("nUser = " + nUser + "\r\n");
		sb.append("nItem = " + nItem + "\r\n");
		sb.append("nRate = " + nRate + "\r\n");
		sb.append("nRate/nUser = " + 1.0F * nRate / nUser + "\r\n");
		sb.append("Sparity = " + (1 - 1.0F * nRate / (nUser * nItem)) + "\r\n");

		FileUtils.write(new File(pivotFile + "-MetaData.dat"), sb.toString(), "",false);
	}

	/**
	 * 汇报基本的统计数据(均值、标准差等)
	 * 
	 * @param rateSet
	 * @throws IOException 
	 */
	public static void reportStatData(RateSet rateSet) throws IOException{
		rateSet.calcMuSigma();

		StringBuffer sb = new StringBuffer();
		sb.append("uid \t nrate \t mu \t sigma\r\n");
		for (int u = 0; u < rateSet.nUser; u++) {
			sb.append(rateSet.getUserId(u) + "\t");
			sb.append(rateSet.getRateList(u).size() + "\t");
			sb.append(rateSet.getMu(u) + "\t" + rateSet.getSigma(u) + "\r\n");
		}
		FileUtils.write(new File(pivotFile + "-USERRateStat.dat"), sb.toString(),"", false);

		RateSet invertRateSet = rateSet.transpose();
		invertRateSet.calcMuSigma();

		sb = new StringBuffer();
		sb.append("iid \t nrate \t mu \t sigma\r\n");
		for (int i = 0; i < rateSet.nItem; i++) {
			sb.append(invertRateSet.getItemId(i) + "\t");
			sb.append(invertRateSet.getRateList(i).size() + "\t");
			sb.append(invertRateSet.getMu(i) + "\t");
			sb.append(invertRateSet.getSigma(i) + "\r\n");
		}
		FileUtils.write(new File(pivotFile + "-ITEMRatedStat.dat"), sb.toString(), "",false);
	}

	/**
	 * 将对称矩阵转成岭形
	 * @throws IOException 
	 */
	public static void ridgeMatrix() throws IOException{
		String file = "data/research/movielens-100k-pivot/";
		String src = file + "Fold1-USERCorateMatrix.dat";
		String dest = file + "Ridge-USERCorateMatrix.dat";

		List<double[]> datum = Data.loadData(src);

		int nLine = datum.size();
		double[][] matrix = new double[nLine][nLine];
		for (int i = 0; i < nLine; i++)
			for (int j = 1; j < nLine; j++)
				matrix[i][j - 1] = datum.get(i)[j];

		matrix = MathLib.Matrix.ridge(matrix);
		StringBuffer content = new StringBuffer();
		for (double[] row : matrix)
			content.append(StringUtils.join(row, "\t") + "\r\n");
		FileUtils.write(new File(dest), content.toString(), "",false);
	}

	public static void main(String[] args) throws IOException{
		TickClock.beginTick();

		int foldId = 1;

		String dbFile = "data/research/movielens-100k/";
		Statistician.pivotFile += "Fold" + foldId;

		String dataFile = dbFile + "u" + foldId + ".base";
		RateSet rateSet = Data.loadRateSet(dataFile);
		Statistician.reportMetaData(rateSet);
		Statistician.reportStatData(rateSet);

		TickClock.stopTick();
	}
}