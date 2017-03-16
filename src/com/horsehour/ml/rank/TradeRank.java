package com.horsehour.ml.rank;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * TradeRank
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140315
 */
public class TradeRank {
	private List<double[]> tradeMatrix;

	public String[] countries;
	public double[] rowSum;
	public double[] colSum;

	public int[] rank;
	public double[] scores;

	public int nNation;
	public int nIter = 500;
	public double epsilon = 1.0E-8;
	public float density;
	public double alpha = 0.5;

	/**
	 * 加载国家列表
	 * 
	 * @param src
	 */
	public void loadCountryList(String src) {
		String line;
		try {
			line = FileUtils.readFileToString(new File(src),"");
			countries = line.trim().split("\r\n");
			nNation = countries.length;
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void loadTradeData(String src) {
		tradeMatrix = Data.loadData(src, "\t");
		rowSum = new double[nNation];
		colSum = new double[nNation];

		for (int i = 0; i < nNation; i++) {
			rowSum[i] = MathLib.Data.sum(tradeMatrix.get(i));
			colSum[i] = MathLib.Data.sum(getColumn(i));
		}
	}

	/**
	 * List<double[]>形式矩阵的列
	 * 
	 * @param rid
	 * @return column of the trade matrix
	 */
	private double[] getColumn(int cid) {
		double[] column = new double[nNation];
		for (int i = 0; i < nNation; i++)
			column[i] = tradeMatrix.get(i)[cid];
		return column;
	}

	private void normMatrix() {
		for (int i = 0; i < nNation; i++) {
			double[] temp = new double[nNation];
			for (int j = 0; j < nNation; j++)
				temp[j] = alpha * tradeMatrix.get(i)[j] / colSum[j]// export
																	// weight
						+ (1 - alpha) * tradeMatrix.get(j)[i] / rowSum[j];// import
																			// weight

			tradeMatrix.set(i, temp);
		}
	}

	/**
	 * 初始化分值
	 */
	private void initScores() {
		scores = new double[nNation];
		for (int i = 0; i < nNation; i++)
			scores[i] = colSum[i];
		MathLib.Scale.max(scores);
	}

	/**
	 * 预处理
	 */
	private void preprocess() {
		int nZero = 0;
		for (int i = 0; i < nNation; i++) {
			if (colSum[i] == 0)
				colSum[i] = 1;

			if (rowSum[i] == 0) {
				rowSum[i] = 1;
				nZero += nNation;
				continue;
			}

			for (double export : tradeMatrix.get(i))
				if (export == 0)
					nZero++;
		}

		density = 1 - (float) (nZero - nNation) / (nNation * (nNation - 1));

		normMatrix();
		initScores();
	}

	/**
	 * 更新分值
	 */
	public double[] updateScores() {
		double[] updated = new double[nNation];
		for (int i = 0; i < nNation; i++)
			updated[i] = MathLib.Matrix.innerProd(scores, tradeMatrix.get(i));
		return updated;
	}

	/**
	 * 根据import与export进行综合排名
	 */
	public void tradeRank() {
		preprocess();

		double residual = epsilon;
		while (0 < nIter-- && residual >= epsilon) {
			double[] weight = updateScores();
			MathLib.Norm.l2(weight);

			residual = MathLib.Distance.euclidean(weight, scores);
			scores = weight;
		}

		MathLib.Matrix.normalize(scores);
		rank = MathLib.getRank(scores, false);
	}

	public void reportRankList(String dest) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < nNation; i++) {
			sb.append((i + 1) + "\t");// 名次
			sb.append(countries[rank[i]] + "\t");
			sb.append(scores[rank[i]] + "\r\n");// 分值
		}
		try {
			FileUtils.write(new File(dest), sb.toString(), "utf-8", false);
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	public void reportRankTrend(String src, String dest) {
		Collection<File> files = FileUtils.listFiles(new File(src), null, false);
		Map<String, String[]> rankList = new HashMap<>();
		int i = 0;
		List<String> lines = null;
		for (File file : files) {
			try {
				lines = FileUtils.readLines(file, "utf-8");
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}

			for (int j = 0; j < lines.size(); j++) {
				String[] meta = lines.get(j).split("\t");
				String nation = meta[1];
				String[] position;
				if (rankList.containsKey(nation))
					position = rankList.get(nation);
				else
					position = new String[18];

				position[i] = meta[0];
				rankList.put(nation, position);
				i++;
			}
		}

		Iterator<Entry<String, String[]>> iter = rankList.entrySet().iterator();
		StringBuffer sb = new StringBuffer();
		while (iter.hasNext()) {
			Entry<String, String[]> entry = iter.next();
			sb.append(entry.getKey());// nation
			for (String position : entry.getValue())
				sb.append("\t" + position);
			sb.append("\r\n");
		}

		try {
			FileUtils.write(new File(dest), sb.toString(), "utf-8", false);
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 根据import进行排名
	 */
	public void importRank() {
		scores = colSum;
		MathLib.Matrix.normalize(scores);
		rank = MathLib.getRank(scores, false);
	}

	/**
	 * 根据export进行排名
	 */
	public void exportRank() {
		scores = rowSum;
		MathLib.Matrix.normalize(scores);
		rank = MathLib.getRank(scores, false);
	}

	/**
	 * 根据import + export进行排名
	 */
	public void ixportRank() {
		scores = MathLib.Matrix.add(rowSum, colSum);
		MathLib.Matrix.normalize(scores);
		rank = MathLib.getRank(scores, false);
	}

	public void reportDegree(String dest) {
		int[] indegree = new int[nNation];
		int[] outdegree = new int[nNation];

		for (int i = 0; i < nNation; i++) {
			double[] export = tradeMatrix.get(i);
			for (int j = 0; j < nNation; j++)
				if (export[j] > 0) {
					outdegree[i] += 1;
					indegree[j] += 1;
				}
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < nNation; i++) {
			sb.append(countries[i] + "\t");
			sb.append(outdegree[i] + "\t");
			sb.append(indegree[i] + "\r\n");
		}

		try {
			FileUtils.write(new File(dest),"", sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String root = "F:/SystemBuilding/Data/WorldTrade/";
		String countryFile = root + "NationCode.txt";

		TradeRank tr = new TradeRank();
		tr.loadCountryList(countryFile);

		double alpha;
		int range = 18;
		for (int i = 0; i < range; i++) {
			tr.loadTradeData(root + (1995 + i) + "-Export.txt");

			// tr.importRank();
			// tr.reportRankList(root + "/RankList/ImportRank/" + (1995 + i) +
			// ".txt");
			//
			// tr.exportRank();
			// tr.reportRankList(root + "/RankList/ExportRank/" + (1995 + i) +
			// ".txt");

			for (int j = 0; j < 5; j++) {
				tr.alpha = alpha = 1.0 - 0.25 * j;
				tr.tradeRank();
				tr.reportRankList(root + "/RankList/TradeRank-alpha=" + alpha + "/" + (1995 + i) + ".txt");
			}

		}

		// tr.reportRankTrend(root + "/RankList/ImportRank/", root +
		// "/RankTrend/ImportRank.txt");
		// tr.reportRankTrend(root + "/RankList/ExportRank/", root +
		// "/RankTrend/ExportRank.txt");

		for (int j = 0; j < 5; j++) {
			alpha = 1.0 - 0.25 * j;
			tr.reportRankTrend(root + "/RankList/TradeRank-alpha=" + alpha + "/",
					root + "/RankTrend/TradeRank-alpha=" + alpha + ".txt");
		}

		TickClock.stopTick();
	}
}
