package com.horsehour.math.opt.lp;

import java.util.Arrays;

import com.horsehour.math.function.Log1pFunction;
import com.horsehour.math.function.TransferFunction;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.TickClock;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;

/**
 * 基于lp_solve求解线性模型
 * 
 * @author Chunheng Jiang
 * @since 20130302
 * @version 1.0
 * @see http://web.mit.edu/lpsolve_v5520/doc/formulate.htm
 * @see http://tech.groups.yahoo.com/group/lp_solve/
 *      http://people.rit.edu/pnveme
 *      /EMEM820n/Mod4_LP/Mod4_content/mod4_sec5_primal_dual.html
 */
public class LPSolve {
	/**
	 * 求解模型
	 * 
	 * @param dir
	 *            : 最值方向
	 * @param objFun
	 *            : 目标函数
	 * @param constraints
	 *            : 约束条件
	 * @param eqType
	 *            : 不等式的形式（le 1,ge 2)
	 * @param bounds
	 *            : 界
	 * @return 模型结果
	 */
	public static Solution solveProblem(ObjDirection dir, double[] objFun,
	        double[][] constraints, int[] eqType, double[] bounds) {
		Solution solution = new Solution();

		int nRow = constraints.length;// 约束的个数
		int nColumn = constraints[0].length;// 未知变量的个数

		// Create Lpsolve linear problem
		LpSolve problem = null;
		try {
			problem = LpSolve.makeLp(0, nColumn);

			if (problem.getLp() != 0) {
				// Set AddRowMode to true
				problem.setAddRowmode(true);

				// Create column index Array (used to add constraints)
				int[] colN = new int[nColumn];
				for (int i = 0; i < nColumn;) {
					colN[i++] = i;
				}

				// Add constraints
				for (int j = 0; j < nRow; j++)
					problem.addConstraintex(nColumn, constraints[j], colN,
					        eqType[j], bounds[j]);

				// Reset Add Row Mode to false
				problem.setAddRowmode(false);

				// Set Objective Function
				problem.setObjFnex(nColumn, objFun, colN);

				// Set Direction
				if (dir == ObjDirection.MAX)
					problem.setMaxim();
				else
					problem.setMinim();

				problem.setVerbose(0);

				// Solve the Linear Problem
				int code = problem.solve();
				if (code == LpSolve.OPTIMAL) {
					solution.obj = problem.getObjective();

					double[] weight = new double[nColumn];
					problem.getVariables(weight);
					solution.vars = weight;
				} else
					return null;
			}
		} catch (LpSolveException e) {
			e.printStackTrace();
			return null;
		}
		if (problem.getLp() != 0)
			problem.deleteLp();

		return solution;
	}

	/**
	 * An example
	 */
	public static void example() {
		double[][] constraints = new double[][] { { 1, 5 }, { 1, 1 } };
		double[] objF = new double[] { 5, 1 };
		double[] rhs = new double[] { 1, 2 };
		ObjDirection dir = ObjDirection.MAX;
		int[] eqType = new int[] { 1, 1 };

		Solution solution = solveProblem(dir, objF, constraints, eqType, rhs);

		System.out.println("Obj:" + solution.obj);
		System.out.print("w:");
		for (double w : solution.vars)
			System.out.print(w + " ");
		System.out.println("");
	}

	/**
	 * 求解DEARank模型
	 * 
	 * @param dataMatrix
	 * @param dir
	 * @return solutions
	 */
	public static Solution[] solveDEARank(SampleSet sampleSet, ObjDirection dir) {
		int nRow = sampleSet.size(), nColumn = sampleSet.dim();

		double[][] dataMatrix = new double[nRow][nColumn];
		populateDataMatrix(sampleSet, dataMatrix);

		int[] eqType = new int[nRow];
		double[] bounds = new double[nRow];
		if (dir == ObjDirection.MAX) {
			Arrays.fill(eqType, 1);// less than
			populateBound(bounds);
		} else {
			Arrays.fill(eqType, 2);// greater than
			populateBound(bounds, sampleSet.getLabels());
		}

		Solution[] solutions = new Solution[dataMatrix.length];
		for (int i = 0; i < nRow; i++) {
			solutions[i] = solveProblem(dir, dataMatrix[i], dataMatrix, eqType,
			        bounds);
		}
		return solutions;
	}

	/**
	 * 填充data matrix
	 * 
	 * @param sampleSet
	 */
	public static void populateDataMatrix(SampleSet sampleSet,
	        double[][] dataMatrix) {
		for (int i = 0; i < dataMatrix.length; i++) {
			for (int j = 0; j < sampleSet.dim(); j++)
				dataMatrix[i][j] = sampleSet.getSample(i).getFeature(j);
		}
	}

	/**
	 * 填充bound
	 * 
	 * @param label
	 */
	public static void populateBound(double[] bounds, Integer[] labels) {
		TransferFunction act = new Log1pFunction((float) Math.E);
		for (int i = 0; i < labels.length; i++)
			bounds[i] = act.calc(labels[i]);
	}

	public static void populateBound(double[] bounds) {
		Arrays.fill(bounds, 1);
	}

	public static void main(String[] args) {
		TickClock.beginTick();
		example();
		TickClock.stopTick();
	}
}