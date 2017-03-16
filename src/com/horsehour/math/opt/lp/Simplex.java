package com.horsehour.math.opt.lp;

/**
 * Simplex Method
 * <p>
 * Given A\in R^{m\times n}, b\in R^m, c\in R^n, 求解线性规划问题：<br/>
 * maximum cx<br/>
 * subject to<br/>
 * Ax <= b, x>=0<br/>
 * 如果b>=0,则x=0就是一个基本可行解.<br/>
 * </p>
 * <p>
 * 创建的单纯形表有m+1行,n+m+1列,其中第m+n列是b(rhs),目标函数在第m行, 从第m列到m+n-1列都是松弛变量.
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130506
 * @see http://algs4.cs.princeton.edu/65reductions/Simplex.java.html
 */
public class Simplex {
	private float epsilon = 1.0E-20F;
	private double[][] tableaux;
	private int nRow;// 约束个数
	private int nCol;// 变量个数
	private int[] basis;// 基元

	public Simplex(double[][] A, double[] b, double[] c) {
		nRow = b.length;
		nCol = c.length;

		tableaux = new double[nRow + 1][nCol + nRow + 1];

		for (int i = 0; i < nRow; i++)
			for (int j = 0; j < nCol; j++)
				tableaux[i][j] = A[i][j];

		for (int i = 0; i < nRow; i++)
			tableaux[i][nCol + i] = 1;

		for (int j = 0; j < nCol; j++)
			tableaux[nRow][j] = c[j];

		for (int i = 0; i < nRow; i++)
			tableaux[i][nRow + nCol] = b[i];

		basis = new int[nRow];
		for (int i = 0; i < nRow; i++)
			basis[i] = nCol + i;
	}

	/**
	 * run simplex algorithm
	 */
	public void solve() {
		while (true) {
			// find entering column q
			int q = bland();
			if (q == -1)
				break; // optimal

			// find leaving row p
			int p = minRatioRule(q);
			if (p == -1)
				throw new RuntimeException("Linear program is unbounded");
			pivot(p, q);
			basis[p] = q;
		}
	}

	/**
	 * @return lowest index of a non-basic column with a positive cost
	 */
	private int bland() {
		for (int j = 0; j < nRow + nCol; j++)
			if (tableaux[nRow][j] > 0)
				return j;
		return -1;
	}

	/**
	 * index of a non-basic column with most positive cost
	 * 
	 * @return
	 */
	public int dantzig() {
		int q = 0;
		for (int j = 1; j < nRow + nCol; j++)
			if (tableaux[nRow][j] > tableaux[nRow][q])
				q = j;

		if (tableaux[nRow][q] <= 0)
			return -1; // optimal
		else
			return q;
	}

	/**
	 * find row p using min ratio rule (-1 if no such row)
	 * 
	 * @param q
	 * @return
	 */
	private int minRatioRule(int q) {
		int p = -1;
		for (int i = 0; i < nRow; i++) {
			if (tableaux[i][q] <= 0)
				continue;
			if (p == -1)
				p = i;
			else if ((tableaux[i][nRow + nCol] / tableaux[i][q]) < (tableaux[p][nRow
			        + nCol] / tableaux[p][q]))
				p = i;
		}
		return p;
	}

	/**
	 * pivot on entry (p, q) using Gauss-Jordan elimination
	 * 
	 * @param p
	 * @param q
	 */
	private void pivot(int p, int q) {
		// everything but row p and column q
		for (int i = 0; i <= nRow; i++)
			for (int j = 0; j <= nRow + nCol; j++)
				if (i != p && j != q)
					tableaux[i][j] -= tableaux[p][j] * tableaux[i][q]
					        / tableaux[p][q];

		// zero out column q
		for (int i = 0; i <= nRow; i++)
			if (i != p)
				tableaux[i][q] = 0;

		// scale row p
		for (int j = 0; j <= nRow + nCol; j++)
			if (j != q)
				tableaux[p][j] /= tableaux[p][q];
		tableaux[p][q] = 1;
	}

	/**
	 * @return optimal objective value
	 */
	public double value() {
		return -tableaux[nRow][nRow + nCol];
	}

	/**
	 * @return primal solution vector
	 */
	public double[] primal() {
		double[] x = new double[nCol];
		for (int i = 0; i < nRow; i++)
			if (basis[i] < nCol)
				x[basis[i]] = tableaux[i][nRow + nCol];
		return x;
	}

	/**
	 * @return dual solution vector
	 */
	public double[] dual() {
		double[] y = new double[nRow];
		for (int i = 0; i < nRow; i++)
			y[i] = -tableaux[nRow][nCol + i];
		return y;
	}

	/**
	 * 判断原问题是否可行
	 * 
	 * @param A
	 * @param b
	 * @return true if primal feasible, false elsewise
	 */
	private boolean isPrimalFeasible(double[][] A, double[] b) {
		double[] x = primal();

		// check that x >= 0
		for (int j = 0; j < x.length; j++) {
			if (x[j] < 0.0) {
				System.out.println("x[" + j + "] = " + x[j] + " is negative");
				return false;
			}
		}

		// check that Ax <= b
		for (int i = 0; i < nRow; i++) {
			double sum = 0.0;
			for (int j = 0; j < nCol; j++) {
				sum += A[i][j] * x[j];
			}
			if (sum > b[i] + epsilon) {
				System.out.println("not primal feasible");
				System.out.println("b[" + i + "] = " + b[i] + ", sum = " + sum);
				return false;
			}
		}
		return true;
	}

	/**
	 * 判断对偶问题是否可行（Dual Feasible）
	 * 
	 * @param A
	 * @param c
	 * @return true if dual feasible, false else
	 */
	private boolean isDualFeasible(double[][] A, double[] c) {
		double[] y = dual();

		// check that y >= 0
		for (int i = 0; i < y.length; i++) {
			if (y[i] < 0.0) {
				System.out.println("y[" + i + "] = " + y[i] + " is negative");
				return false;
			}
		}

		// check that yA >= c
		for (int j = 0; j < nCol; j++) {
			double sum = 0.0;
			for (int i = 0; i < nRow; i++) {
				sum += A[i][j] * y[i];
			}
			if (sum < c[j] - epsilon) {
				System.out.println("not dual feasible");
				System.out.println("c[" + j + "] = " + c[j] + ", sum = " + sum);
				return false;
			}
		}
		return true;
	}

	// check that optimal value = cx = yb
	private boolean isOptimal(double[] b, double[] c) {
		double[] x = primal();
		double[] y = dual();
		double value = value();

		// check that value = cx = yb
		double value1 = 0;
		for (int j = 0; j < x.length; j++)
			value1 += c[j] * x[j];
		double value2 = 0;
		for (int i = 0; i < y.length; i++)
			value2 += y[i] * b[i];
		if (Math.abs(value - value1) > epsilon
		        || Math.abs(value - value2) > epsilon) {
			System.out.println("value = " + value + ", cx = " + value1
			        + ", yb = " + value2);
			return false;
		}
		return true;
	}

	/**
	 * 检查最优性条件(optimality conditions)
	 * 
	 * @param A
	 * @param b
	 * @param c
	 * @return 通过则true，否则为false
	 */
	public boolean check(double[][] A, double[] b, double[] c) {
		return isPrimalFeasible(A, b) && isDualFeasible(A, c)
		        && isOptimal(b, c);
	}

	/**
	 * 打印单纯形表
	 */
	public void show() {
		System.out.println("M = " + nRow);
		System.out.println("N = " + nCol);
		for (int i = 0; i <= nRow; i++) {
			for (int j = 0; j <= nRow + nCol; j++)
				System.out.printf("%7.2f ", tableaux[i][j]);
			System.out.println();
		}

		System.out.println("value = " + value());
		for (int i = 0; i < nRow; i++)
			if (basis[i] < nCol)
				System.out.println("x_" + basis[i] + " = "
				        + tableaux[i][nRow + nCol]);
		System.out.println();
	}

	public void setEpsilon(float eps) {
		this.epsilon = eps;
	}
}