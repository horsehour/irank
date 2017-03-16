package com.horsehour.ml.rank;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.metric.NDCG;
import com.horsehour.ml.metric.Precision;
import com.horsehour.util.TickClock;

/**
 * 使用CCR模型计算的相对效率值给文档打分
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130509
 */
public class CCRRank {
	private final Metric[] metrics;

	public CCRRank() {
		int m = 11;
		metrics = new Metric[m];
		metrics[5] = new MAP();

		for (int k = 0; k < 5; k++) {
			metrics[k] = new Precision(k + 1);
			metrics[k + 6] = new NDCG(k + 1);
		}
	}

	/**
	 * 根据name模型计算各个DMU的相对效率分值,使用效率分值作为排名基准,衡量排名性能
	 * 
	 * @param name
	 */
	public void evaluate(String name){
		String src = "";
		String dest = "";
		String root = "F:/Research/Data/CCRData/";

		String[] corpusList = {"MQ2007", "MQ2008", "OHSUMED"};

		List<double[]> datum;
		for (String corpus : corpusList) {
			dest = "F:/Research/Experiments/20130609" + corpus + ".eval";
			for (int i = 1; i <= 5; i++) {
				src = root + corpus + "/Fold" + i + "/" + name + ".txt";
				datum = Data.loadData(src, "utf-8", "	");
				archiveEvaluation(evaluate(datum), name, dest);
			}
		}
	}

	/**
	 * 使用多个指标度量
	 * 
	 * @param dataset
	 * @return evaluation based on metrics
	 */
	private double[] evaluate(List<double[]> datum){
		double[] perfs = new double[metrics.length];

		List<Integer> labelList = null;
		List<Double> scoreList = null;

		int qid = -1, preqid = 0;
		int count = 0;
		for (int i = 0; i < datum.size(); i++) {
			double[] lineData = datum.get(i);
			qid = (int) lineData[2];
			if (preqid != qid) {
				count++;
				// Evaluating
				if (labelList != null)
					for (int j = 0; j < perfs.length; j++)
						perfs[j] += metrics[j].measure(labelList, scoreList);

				labelList = new ArrayList<Integer>();
				scoreList = new ArrayList<Double>();

				preqid = qid;
			}
			labelList.add((int) lineData[0]);
			scoreList.add(lineData[1]);
		}
		for (int k = 0; k < metrics.length; k++)
			perfs[k] /= count;

		return perfs;
	}

	/**
	 * 将评估结果保存到文件中
	 * 
	 * @param eval
	 * @param dest
	 */
	private void archiveEvaluation(double[] eval, String model, String dest){
		StringBuffer sb = new StringBuffer();
		sb.append(model);

		for (int i = 0; i < eval.length; i++)
			sb.append("\t" + eval[i]);
		sb.append("\r\n");
		try {
			FileUtils.write(new File(dest), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	public static void main(String[] args){
		TickClock.beginTick();

		CCRRank ranker = new CCRRank();

		String[] models = {"Test-IC2R", "Test-OC2R"};
		for (int i = 0; i < models.length; i++)
			ranker.evaluate(models[i]);

		TickClock.stopTick();
	}
}