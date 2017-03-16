package com.horsehour.ml.rank.weak;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;

import org.apache.commons.io.FileUtils;

import com.horsehour.math.function.Log1pFunction;
import com.horsehour.math.function.TransferFunction;
import com.horsehour.math.opt.lp.ObjDirection;
import com.horsehour.math.opt.lp.Solution;
import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.sieve.L2RSieve;

/**
 * 使用lp_solve包求解CCR模型,将结果保存到本地文件-CCRData
 * <p>
 * 根据特殊的CCR模型,计算各个决策单元的相对最优权值,并保存到本地文件
 * </p>
 * <p>
 * 特别地,应用到排序学习问题,计算训练数据集中所有检索词-文档对的最优权值并保存
 * </p>
 * <p>
 * 数据保存格式与l2r原始数据类似，格式如下<br/>
 * 1 0.89 902 0.29 0.03 ...<br/>
 * 0 0.38 902 0.04 0.09 ...<br/>
 * 第一列表示文档的真实评分,第二列表示相对权值,第三列表示检索词id,后续各列表示相应的最优权值向量
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130508
 * @see http://web.mit.edu/lpsolve_v5520/java/docs/api/index-all.html
 */

public class WeakGenerator {
	private DataSet trainset;
	private String dest = "";

	private int oriented = 0;

	// 直接使用lp_solve求解ccr模型
	public LpSolve problem;
	public double[][] dataMatrix;
	ObjDirection dir = ObjDirection.MAX;
	public TransferFunction activeFun = new Log1pFunction((float) Math.E);

	public double[] bound;
	public int eqType = -1;

	public int nRow = -1, nCol = -1;

	public WeakGenerator() {}

	/**
	 * 基于lp_solve求解CCR模型
	 */
	public void solveCCR(){
		dir = (oriented == 0) ? ObjDirection.MIN : ObjDirection.MAX;

		SampleSet sampleSet;
		Solution[] sols;

		StringBuffer sb;
		for (int qid = 0; qid < trainset.size(); qid++) {
			sampleSet = trainset.getSampleSet(qid);
			nRow = sampleSet.size();

			prepareData(sampleSet);
			makeProblem();

			String query = sampleSet.getSample(0).getQid();
			sb = new StringBuffer();
			sols = solveProblem();

			for (int id = 0; id < sols.length; id++) {
				if (sols[id] == null)
					continue;

				sb.append(sampleSet.getLabel(id));
				sb.append("\t" + sols[id].obj);
				sb.append("\t" + query);

				for (double weight : sols[id].vars)
					sb.append("\t" + weight);

				sb.append("\r\n");
			}
			try {
				FileUtils.write(new File(dest), sb.toString(),"");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * 根据模型从sampleset中配置必要的数据 1. dataMatrix 2. bound 3. eqType
	 * 
	 * @param sampleset
	 */
	private void prepareData(SampleSet sampleset){
		dataMatrix = new double[nRow][nCol];
		for (int i = 0; i < nRow; i++)
			for (int j = 0; j < nCol; j++)
				dataMatrix[i][j] = sampleset.getSample(i).getFeature(j);

		bound = new double[nRow];
		// 根据目标函数的类型设置约束条件(不等式符号,界)
		if (dir == ObjDirection.MAX) {
			eqType = 1;// leq
			Arrays.fill(bound, 1);
		} else {
			eqType = 2;// geq

			List<Integer> labels = sampleset.getLabelList();
			for (int i = 0; i < nRow; i++)
				bound[i] = activeFun.calc(labels.get(i));
		}
	}

	/**
	 * 构造模型-problem
	 */
	private void makeProblem(){
		try {
			problem = LpSolve.makeLp(0, nCol);

			// 设置目标函数类型
			if (dir == ObjDirection.MAX)
				problem.setMaxim();
			else
				problem.setMinim();

			if (problem.getLp() != 0) {
				problem.setAddRowmode(true);
				// 添加约束条件
				for (int i = 0; i < nRow; i++)
					problem.addConstraint(dataMatrix[i], eqType, bound[i]);

				problem.setAddRowmode(false);
				problem.setVerbose(0);// 不打印日志
			}
		} catch (LpSolveException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 求解模型
	 * 
	 * @return 问题的解
	 */
	private Solution[] solveProblem(){
		Solution[] sols = new Solution[nRow];

		double[] weight = new double[nCol];
		for (int i = 0; i < nRow; i++) {
			try {
				// 设置目标函数
				problem.setObjFn(dataMatrix[i]);

				// 求解模型
				int code = problem.solve();
				if (code == LpSolve.OPTIMAL) {
					problem.getVariables(weight);
					sols[i] = new Solution(problem.getObjective(), weight);
				}

			} catch (LpSolveException e) {
				e.printStackTrace();
				return null;
			}
		}

		if (problem.getLp() != 0)
			problem.deleteLp();

		return sols;
	}

	public void solve(String workspace, String ori){
		trainset = Data.loadDataSet(workspace + "/train.txt", new L2RSieve());
		nCol = trainset.dim();

		if (ori.equalsIgnoreCase("O")) {
			oriented = 0;
			dest = workspace + "OLPDEA.weak";
		} else if (ori.equalsIgnoreCase("I")) {
			oriented = 1;
			dest = workspace + "ILPDEA.weak";
		}

		solveCCR();
	}
}
