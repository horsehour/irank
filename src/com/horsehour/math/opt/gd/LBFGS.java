package com.horsehour.math.opt.gd;

/**
 * Approximation of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using
 * a limited amount of computer memory
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2014年5月2日 上午8:26:28
 **/
public class LBFGS {
	/**
	 * Controls the accuracy of the line search <code>mcsrch</code>. If the
	 * function and gradient evaluations are inexpensive with respect to the
	 * cost of the iteration (which is sometimes the case when solving very
	 * large problems) it may be advantageous to set <code>gtol</code> to a
	 * small value. A typical small value is 0.1. Restriction: <code>gtol</code>
	 * should be greater than 1e-4.
	 */

	public static double gtol = 0.9;

	/**
	 * Specify lower bound for the step in the line search. The default value is
	 * 1e-20. This value need not be modified unless the exponent is too large
	 * for the machine being used, or unless the problem is extremely badly
	 * scaled (in which case the exponent should be increased).
	 */

	public static double stpmin = 1E-20;

	/**
	 * Specify upper bound for the step in the line search. The default value is
	 * 1e20. This value need not be modified unless the exponent is too large
	 * for the machine being used, or unless the problem is extremely badly
	 * scaled (in which case the exponent should be increased).
	 */

	public static double stpmax = 1E20;

	public static int maxfev = 200;

	/**
	 * The solution vector as it was at the end of the most recently completed
	 * line search. This will usually be different from the return value of the
	 * parameter <tt>x</tt> of <tt>lbfgs</tt>, which is modified by line-search
	 * steps. A caller which wants to stop the optimization iterations before
	 * <tt>LBFGS.lbfgs</tt> automatically stops (by reaching a very small
	 * gradient) should copy this vector instead of using <tt>x</tt>. When
	 * <tt>LBFGS.lbfgs</tt> automatically stops, then <tt>x</tt> and
	 * <tt>solution_cache</tt> are the same.
	 */
	public static double[] solution_cache = null;

	private static double gnorm = 0, stp1 = 0, ftol = 0, stp[] = new double[1],
	        ys = 0, yy = 0, sq = 0, yr = 0, beta = 0, xnorm = 0;
	private static int iter = 0, nfun = 0, point = 0, ispt = 0, iypt = 0,
	        info[] = new int[1], bound = 0, npt = 0, cp = 0, i = 0,
	        nfev[] = new int[1], inmc = 0, iycn = 0, iscn = 0;
	private static boolean finish = false;

	private static double[] w = null;

	/**
	 * This method returns the total number of evaluations of the objective
	 * function since the last time LBFGS was restarted. The total number of
	 * function evaluations increases by the number of evaluations required for
	 * the line search; the total is only increased after a successful line
	 * search.
	 */
	public static int nfevaluations() {
		return nfun;
	}

	/**
	 * This subroutine solves the unconstrained minimization problem
	 * 
	 * <pre>
	 *     min f(x),    x = (x1,x2,...,x_n),
	 * </pre>
	 * 
	 * using the limited-memory BFGS method. The routine is especially effective
	 * on problems involving a large number of variables. In a typical iteration
	 * of this method an approximation <code>Hk</code> to the inverse of the
	 * Hessian is obtained by applying <code>m</code> BFGS updates to a diagonal
	 * matrix <code>Hk0</code>, using information from the previous M steps. The
	 * user specifies the number <code>m</code>, which determines the amount of
	 * storage required by the routine. The user may also provide the diagonal
	 * matrices <code>Hk0</code> if not satisfied with the default choice. The
	 * algorithm is described in "On the limited memory BFGS method for large
	 * scale optimization", by D. Liu and J. Nocedal, Mathematical Programming B
	 * 45 (1989) 503-528.
	 * 
	 * The user is required to calculate the function value <code>f</code> and
	 * its gradient <code>g</code>. In order to allow the user complete control
	 * over these computations, reverse communication is used. The routine must
	 * be called repeatedly under the control of the parameter
	 * <code>iflag</code>.
	 * 
	 * The steplength is determined at each iteration by means of the line
	 * search routine <code>mcsrch</code>, which is a slight modification of the
	 * routine <code>CSRCH</code> written by More' and Thuente.
	 * 
	 * The only variables that are machine-dependent are <code>xtol</code>,
	 * <code>stpmin</code> and <code>stpmax</code>.
	 * 
	 * Progress messages and non-fatal error messages are printed to
	 * <code>System.err</code>. Fatal errors cause exception to be thrown, as
	 * listed below.
	 * 
	 * @param n
	 *            The number of variables in the minimization problem.
	 *            Restriction: <code>n &gt; 0</code>.
	 * 
	 * @param m
	 *            The number of corrections used in the BFGS update. Values of
	 *            <code>m</code> less than 3 are not recommended; large values
	 *            of <code>m</code> will result in excessive computing time.
	 *            <code>3 &lt;= m &lt;= 7</code> is recommended. Restriction:
	 *            <code>m &gt; 0</code>.
	 * 
	 * @param x
	 *            On initial entry this must be set by the user to the values of
	 *            the initial estimate of the solution vector. On exit with
	 *            <code>iflag = 0</code>, it contains the values of the
	 *            variables at the best point found (usually a solution).
	 * 
	 * @param f
	 *            Before initial entry and on a re-entry with
	 *            <code>iflag = 1</code>, it must be set by the user to contain
	 *            the value of the function <code>f</code> at the point
	 *            <code>x</code>.
	 * 
	 * @param g
	 *            Before initial entry and on a re-entry with
	 *            <code>iflag = 1</code>, it must be set by the user to contain
	 *            the components of the gradient <code>g</code> at the point
	 *            <code>x</code>.
	 * 
	 * @param diagco
	 *            Set this to <code>true</code> if the user wishes to provide
	 *            the diagonal matrix <code>Hk0</code> at each iteration.
	 *            Otherwise it should be set to <code>false</code> in which case
	 *            <code>lbfgs</code> will use a default value described below.
	 *            If <code>diagco</code> is set to <code>true</code> the routine
	 *            will return at each iteration of the algorithm with
	 *            <code>iflag = 2</code>, and the diagonal matrix
	 *            <code>Hk0</code> must be provided in the array
	 *            <code>diag</code>.
	 * 
	 * @param diag
	 *            If <code>diagco = true</code>, then on initial entry or on
	 *            re-entry with <code>iflag = 2</code>, <code>diag</code> must
	 *            be set by the user to contain the values of the diagonal
	 *            matrix <code>Hk0</code>. Restriction: all elements of
	 *            <code>diag</code> must be positive.
	 * 
	 * @param iprint
	 *            Specifies output generated by <code>lbfgs</code>.
	 *            <code>iprint[0]</code> specifies the frequency of the output:
	 *            <ul>
	 *            <li> <code>iprint[0] &lt; 0</code>: no output is generated,
	 *            <li> <code>iprint[0] = 0</code>: output only at first and last
	 *            iteration,
	 *            <li> <code>iprint[0] &gt; 0</code>: output every
	 *            <code>iprint[0]</code> iterations.
	 *            </ul>
	 * 
	 *            <code>iprint[1]</code> specifies the type of output generated:
	 *            <ul>
	 *            <li> <code>iprint[1] = 0</code>: iteration count, number of
	 *            function evaluations, function value, norm of the gradient,
	 *            and steplength,
	 *            <li> <code>iprint[1] = 1</code>: same as
	 *            <code>iprint[1]=0</code>, plus vector of variables and
	 *            gradient vector at the initial point,
	 *            <li> <code>iprint[1] = 2</code>: same as
	 *            <code>iprint[1]=1</code>, plus vector of variables,
	 *            <li> <code>iprint[1] = 3</code>: same as
	 *            <code>iprint[1]=2</code>, plus gradient vector.
	 *            </ul>
	 * 
	 * @param eps
	 *            Determines the accuracy with which the solution is to be
	 *            found. The subroutine terminates when
	 * 
	 *            <pre>
	 *            ||G|| &lt; EPS max(1,||X||),
	 * </pre>
	 * 
	 *            where <code>||.||</code> denotes the Euclidean norm.
	 * 
	 * @param xtol
	 *            An estimate of the machine precision (e.g. 10e-16 on a SUN
	 *            station 3/60). The line search routine will terminate if the
	 *            relative width of the interval of uncertainty is less than
	 *            <code>xtol</code>.
	 * 
	 * @param iflag
	 *            This must be set to 0 on initial entry to <code>lbfgs</code>.
	 *            A return with <code>iflag &lt; 0</code> indicates an error,
	 *            and <code>iflag = 0</code> indicates that the routine has
	 *            terminated without detecting errors. On a return with
	 *            <code>iflag = 1</code>, the user must evaluate the function
	 *            <code>f</code> and gradient <code>g</code>. On a return with
	 *            <code>iflag = 2</code>, the user must provide the diagonal
	 *            matrix <code>Hk0</code>.
	 * 
	 *            The following negative values of <code>iflag</code>, detecting
	 *            an error, are possible:
	 *            <ul>
	 *            <li> <code>iflag = -1</code> The line search routine
	 *            <code>mcsrch</code> failed. One of the following messages is
	 *            printed:
	 *            <ul>
	 *            <li>Improper input parameters.
	 *            <li>Relative width of the interval of uncertainty is at most
	 *            <code>xtol</code>.
	 *            <li>More than 20 function evaluations were required at the
	 *            present iteration.
	 *            <li>The step is too small.
	 *            <li>The step is too large.
	 *            <li>Rounding errors prevent further progress. There may not be
	 *            a step which satisfies the sufficient decrease and curvature
	 *            conditions. Tolerances may be too small.
	 *            </ul>
	 *            <li><code>iflag = -2</code> The i-th diagonal element of the
	 *            diagonal inverse Hessian approximation, given in DIAG, is not
	 *            positive.
	 *            <li><code>iflag = -3</code> Improper input parameters for
	 *            LBFGS (<code>n</code> or <code>m</code> are not positive).
	 *            </ul>
	 * @throws Exception
	 * @throws LBFGS.ExceptionWithIflag
	 */

	public static void lbfgs(int n, int m, double[] x, double f, double[] g,
	        boolean diagco, double[] diag, int[] iprint, double eps,
	        double xtol, int[] iflag) throws Exception {

		boolean execute_entire_while_loop = false;
		if (w == null || w.length != n * (2 * m + 1) + 2 * m) {
			w = new double[n * (2 * m + 1) + 2 * m];
		}

		if (iflag[0] == 0) {
			// Initialize.
			solution_cache = new double[n];
			System.arraycopy(x, 0, solution_cache, 0, n);

			iter = 0;

			if (n <= 0 || m <= 0) {
				iflag[0] = -3;
				throw new Exception(
				        "Improper input parameters  (n or m are not positive.)");
			}

			if (gtol <= 0.0001) {
				System.err
				        .println("LBFGS.lbfgs: gtol is less than or equal to 0.0001. "
				                + "It has been reset to 0.9.");
				gtol = 0.9;
			}

			nfun = 1;
			point = 0;
			finish = false;

			if (diagco) {
				for (i = 1; i <= n; i += 1) {
					if (diag[i - 1] <= 0) {
						iflag[0] = -2;
						throw new Exception(
						        "The "
						                + i
						                + "-th diagonal element of "
						                + "the inverse hessian approximation is not positive.");
					}
				}
			} else {
				for (i = 1; i <= n; i += 1) {
					diag[i - 1] = 1;
				}
			}
			ispt = n + 2 * m;
			iypt = ispt + n * m;

			for (i = 1; i <= n; i += 1) {
				w[ispt + i - 1] = -g[i - 1] * diag[i - 1];
				if (badNum(w[ispt + i - 1]))
					System.err.println("w1 " + g[i - 1] + " " + diag[i - 1]);
			}

			gnorm = Math.sqrt(ddot(n, g, 0, 1, g, 0, 1));
			stp1 = 1 / gnorm;
			ftol = 0.0001;

			if (iprint[1 - 1] >= 0)
				lb1(iprint, iter, nfun, gnorm, n, m, x, f, g, stp, finish);

			execute_entire_while_loop = true;
		}

		while (true) {
			if (execute_entire_while_loop) {
				iter = iter + 1;
				info[0] = 0;
				bound = iter - 1;
				if (iter != 1) {
					if (iter > m)
						bound = m;
					ys = ddot(n, w, iypt + npt, 1, w, ispt + npt, 1);
					if (!diagco) {
						yy = ddot(n, w, iypt + npt, 1, w, iypt + npt, 1);

						for (i = 1; i <= n; i += 1) {
							diag[i - 1] = safeDiv(ys, yy);
						}
					} else {
						iflag[0] = 2;
						return;
					}
				}
			}

			if (execute_entire_while_loop || iflag[0] == 2) {
				if (iter != 1) {
					if (diagco) {
						for (i = 1; i <= n; i += 1) {
							if (diag[i - 1] <= 0) {
								iflag[0] = -2;
								throw new Exception(
								        "The "
								                + i
								                + "-th diagonal element "
								                + "of the inverse hessian approximation is not positive.");
							}
						}
					}
					cp = point;
					if (point == 0)
						cp = m;
					w[n + cp - 1] = 1 / ys;
					if (badNum(w[n + cp - 1]))
						System.err.println("w2 " + ys);

					for (i = 1; i <= n; i += 1) {
						w[i - 1] = -g[i - 1];
						if (badNum(w[i - 1]))
							throw new IllegalArgumentException(i + " "
							        + g[i - 1]);
					}

					cp = point;

					for (i = 1; i <= bound; i += 1) {
						cp = cp - 1;
						if (cp == -1)
							cp = m - 1;
						sq = ddot(n, w, ispt + cp * n, 1, w, 0, 1);
						inmc = n + m + cp + 1;
						iycn = iypt + cp * n;
						w[inmc - 1] = w[n + cp] * sq;
						if (badNum(w[inmc - 1])) {
							System.err.println("w3 " + w[n + cp] + " " + sq);
							lb1(iprint, iter, nfun, gnorm, n, m, x, f, g, stp,
							        finish);
						}
						daxpy(n, -w[inmc - 1], w, iycn, 1, w, 0, 1);
					}

					for (i = 1; i <= n; i += 1) {
						w[i - 1] *= diag[i - 1];
						if (badNum(w[i - 1]))
							System.err.println("w4 " + diag[i - 1]);
					}

					for (i = 1; i <= bound; i += 1) {
						yr = ddot(n, w, iypt + cp * n, 1, w, 0, 1);
						beta = safeMult(w[n + cp], yr);
						inmc = n + m + cp + 1;
						beta = w[inmc - 1] - beta;
						if (badNum(beta)) {
							throw new IllegalArgumentException(w[inmc - 1]
							        + " " + safeMult(w[n + cp], yr));
						}
						iscn = ispt + cp * n;
						daxpy(n, beta, w, iscn, 1, w, 0, 1);
						cp = cp + 1;
						if (cp == m)
							cp = 0;
					}

					for (i = 1; i <= n; i += 1) {
						w[ispt + point * n + i - 1] = w[i - 1];
						if (badNum(w[ispt + point * n + i - 1]))
							System.err.println("w5 " + w[i - 1]);
					}
				}

				nfev[0] = 0;

				LBFGS.Mcsrch.setSTP(stp, iter == 1 && stp1 < 1 ? stp1 : 1);

				for (i = 1; i <= n; i += 1) {
					w[i - 1] = g[i - 1];
					if (badNum(w[i - 1]))
						System.err.println("w6 " + g[i - 1]);
				}
			}
			double MY_MAX_STEP = 1;
			double directionMag = ddot(n, w, ispt + point * n, 1, w, ispt
			        + point * n, 1);
			double effectiveStep = stp[0] * directionMag;
			if (effectiveStep > MY_MAX_STEP) {
				if (iprint[0] > 0)
					System.err.println("reducing step: " + stp[0] + " => "
					        + (MY_MAX_STEP / directionMag));
			} else
				stpmax = 1e20;

			Mcsrch.mcsrch(n, x, f, g, w, ispt + point * n, stp, ftol, xtol,
			        maxfev, info, nfev, diag, iprint);

			if (info[0] == -1) {
				iflag[0] = 1;
				return;
			}

			if (iprint[0] > 0)
				System.err.println("msrch return = nfev=" + nfev[0] + " nfun="
				        + nfun + " info=" + info[0] + " bestx=" + Mcsrch.stx[0]
				        + " farx=" + Mcsrch.sty[0] + " brackt="
				        + Mcsrch.brackt[0] + " stp=" + stp[0] + " gnorm="
				        + Math.sqrt(ddot(n, g, 0, 1, g, 0, 1)) + " xnorm="
				        + Math.sqrt(ddot(n, x, 0, 1, x, 0, 1)));

			if (info[0] != 1) {
				iflag[0] = -1;
				throw new Exception(
				        "Line search failed. See documentation of routine mcsrch. "
				                + "Error return of line search: info = "
				                + info[0]
				                + " Possible causes: function or gradient are "
				                + "incorrect, or incorrect tolerances.");
			}

			nfun += nfev[0];
			npt = point * n;

			for (i = 1; i <= n; i += 1) {
				w[ispt + npt + i - 1] *= stp[0];
				if (badNum(w[ispt + npt + i - 1]))
					System.err.println("w6 " + stp[0]);
				w[iypt + npt + i - 1] = g[i - 1] - w[i - 1];
				if (badNum(w[iypt + npt + i - 1]))
					System.err.println("w7 " + g[i - 1] + " " + w[i - 1]);
			}

			point++;
			if (point == m)
				point = 0;

			gnorm = Math.sqrt(ddot(n, g, 0, 1, g, 0, 1));
			xnorm = Math.sqrt(ddot(n, x, 0, 1, x, 0, 1));
			xnorm = Math.max(1.0, xnorm);

			if (gnorm / xnorm <= eps)
				finish = true;

			if (iprint[1 - 1] >= 0)
				lb1(iprint, iter, nfun, gnorm, n, m, x, f, g, stp, finish);

			// Cache the current solution vector. Due to the spaghetti-like
			// nature of this code, it's not possible to quit here and return;
			// we need to go back to the top of the loop, and eventually call
			// mcsrch one more time -- but that will modify the solution vector.
			// So we need to keep a copy of the solution vector as it was at
			// the completion (info[0]==1) of the most recent line search.
			System.arraycopy(x, 0, solution_cache, 0, n);

			if (finish) {
				iflag[0] = 0;
				return;
			}

			execute_entire_while_loop = true; // from now on, execute whole loop
		}
	}

	/**
	 * Print debugging and status messages for <code>lbfgs</code>. Depending on
	 * the parameter <code>iprint</code>, this can include number of function
	 * evaluations, current function value, etc. The messages are output to
	 * <code>System.err</code>.
	 * 
	 * @param iprint
	 *            Specifies output generated by <code>lbfgs</code>.
	 *            <p>
	 *            <code>iprint[0]</code> specifies the frequency of the output:
	 *            <ul>
	 *            <li> <code>iprint[0] &lt; 0</code>: no output is generated,
	 *            <li> <code>iprint[0] = 0</code>: output only at first and last
	 *            iteration,
	 *            <li> <code>iprint[0] &gt; 0</code>: output every
	 *            <code>iprint[0]</code> iterations.
	 *            </ul>
	 *            <p>
	 * 
	 *            <code>iprint[1]</code> specifies the type of output generated:
	 *            <ul>
	 *            <li> <code>iprint[1] = 0</code>: iteration count, number of
	 *            function evaluations, function value, norm of the gradient,
	 *            and steplength,
	 *            <li> <code>iprint[1] = 1</code>: same as
	 *            <code>iprint[1]=0</code>, plus vector of variables and
	 *            gradient vector at the initial point,
	 *            <li> <code>iprint[1] = 2</code>: same as
	 *            <code>iprint[1]=1</code>, plus vector of variables,
	 *            <li> <code>iprint[1] = 3</code>: same as
	 *            <code>iprint[1]=2</code>, plus gradient vector.
	 *            </ul>
	 * @param iter1
	 *            Number of iterations so far.
	 * @param nfun1
	 *            Number of function evaluations so far.
	 * @param gnorm1
	 *            Norm of gradient at current solution <code>x</code>.
	 * @param n
	 *            Number of free parameters.
	 * @param m
	 *            Number of corrections kept.
	 * @param x
	 *            Current solution.
	 * @param f
	 *            Function value at current solution.
	 * @param g
	 *            Gradient at current solution <code>x</code>.
	 * @param stp11
	 *            Current stepsize.
	 * @param finish1
	 *            Whether this method should print the ``we're done'' message.
	 */
	public static void lb1(int[] iprint, int iter1, int nfun1, double gnorm1,
	        int n, int m, double[] x, double f, double[] g, double[] stp11,
	        boolean finish1) {
		String heading = "\ti\tnfn\tfunc\t\t\tgnorm\t\t\tsteplength";
		int i1;

		if (iter1 == 0) {
			System.err
			        .println("*************************************************");
			System.err.println("  n = " + n + "   number of corrections = " + m
			        + "\n       initial values");
			System.err.println(" f =  " + f + "   gnorm =  " + gnorm1);
			if (iprint[2 - 1] >= 1) {
				System.err.print(" vector x =");
				for (i1 = 1; i1 <= n; i1++)
					System.err.print("  " + x[i1 - 1]);
				System.err.println("");

				System.err.print(" gradient vector g =");
				for (i1 = 1; i1 <= n; i1++)
					System.err.print("  " + g[i1 - 1]);
				System.err.println("");
			}
			System.err
			        .println("*************************************************");
			System.err.println(heading);
		} else {
			if ((iprint[1 - 1] == 0) && (iter1 != 1 && !finish1))
				return;
			if (iprint[1 - 1] != 0) {
				if ((iter1 - 1) % iprint[1 - 1] == 0 || finish1) {
					if (iprint[2 - 1] > 1 && iter1 > 1)
						System.err.println(heading);
					System.err.println("\t" + iter1 + "\t" + nfun1 + "\t" + f
					        + "\t" + gnorm1 + "\t" + stp11[0]);
				} else {
					return;
				}
			} else {
				if (iprint[2 - 1] > 1 && finish1)
					System.err.println(heading);
				System.err.println("\t" + iter1 + "\t" + nfun1 + "\t" + f
				        + "\t" + gnorm1 + "\t" + stp11[0]);
			}
			if (iprint[2 - 1] == 2 || iprint[2 - 1] == 3) {
				if (finish1) {
					System.err.print(" final point x =");
				} else {
					System.err.print(" vector x =  ");
				}
				for (i1 = 1; i1 <= n; i1++)
					System.err.print("  " + x[i1 - 1]);
				System.err.println("");
				if (iprint[2 - 1] == 3) {
					System.err.print(" gradient vector g =");
					for (i1 = 1; i1 <= n; i1++)
						System.err.print("  " + g[i1 - 1]);
					System.err.println("");
				}
			}
			if (finish1)
				System.err
				        .println(" The minimization terminated without detecting errors. iflag = 0");
		}
		return;
	}

	/**
	 * Compute the sum of a vector times a scalara plus another vector. Adapted
	 * from the subroutine <code>daxpy</code> in <code>lbfgs.f</code>. There
	 * could well be faster ways to carry out this operation; this code is a
	 * straight translation from the Fortran.
	 */
	public static void daxpy(int n, double da, double[] dx, int ix0, int incx,
	        double[] dy, int iy0, int incy) {
		int i1, ix, iy;// , m, mp1;

		if (n <= 0)
			return;

		if (da == 0)
			return;

		// if (!(incx == 1 && incy == 1)) {
		ix = 1;
		iy = 1;

		if (incx < 0)
			ix = (-n + 1) * incx + 1;
		if (incy < 0)
			iy = (-n + 1) * incy + 1;

		for (i1 = 1; i1 <= n; i1 += 1) {
			dy[iy0 + iy - 1] += safeMult(da, dx[ix0 + ix - 1]);
			ix = ix + incx;
			iy = iy + incy;
		}

		return;
	}

	/**
	 * Compute the dot product of two vectors. Adapted from the subroutine
	 * <code>ddot</code> in <code>lbfgs.f</code>. There could well be faster
	 * ways to carry out this operation; this code is a straight translation
	 * from the Fortran.
	 */
	static double ddot(int n, double[] dx, int ix0, int incx, double[] dy,
	        int iy0, int incy) {
		double result = 0;
		for (int in = 0, ix = ix0, iy = iy0; in < n; in++, ix += incx, iy += incy) {
			result += safeMult(dx[ix], dy[iy]);
		}
		return result;
	}

	static boolean badNum(double n) {
		return Double.isNaN(n) || Double.isInfinite(n);
	}

	static double safeMult(double n1, double n2) {
		double result = n1 * n2;
		if (badNum(result)) {
			if (n1 == 0 || n2 == 0) {
				result = 0;
			} else {
				throw new IllegalArgumentException(n1 + " " + n2);
			}
		}
		return result;
	}

	public static double safeDiv(double n1, double n2) {
		double result = n1 / n2;
		if (badNum(result)) {
			if (n1 == 0 && n2 == 0) {
				result = 1;
			} else {
				throw new IllegalArgumentException(n1 + " " + n2);
			}
		}
		return result;
	}

	/**
	 * This class implements an algorithm for multi-dimensional line search.
	 * This file is a translation of Fortran code written by Jorge Nocedal. See
	 * comments in the file <tt>LBFGS.java</tt> for more information.
	 */
	public static class Mcsrch {
		private static int infoc[] = new int[1], j = 0;
		private static double dg = 0, dgm = 0, dginit = 0, dgtest = 0,
		        dgx[] = new double[1], dgxm[] = new double[1],
		        dgy[] = new double[1], dgym[] = new double[1], finit = 0,
		        ftest1 = 0, fm = 0, fx[] = new double[1],
		        fxm[] = new double[1], fy[] = new double[1],
		        fym[] = new double[1];
		static double stx[] = new double[1];
		static double sty[] = new double[1];
		static double stmin = 0, stmax = 0, width = 0, width1 = 0;
		static boolean brackt[] = new boolean[1];
		static boolean stage1 = false;
		private static final double ONE_HALF = 0.5, TWO_THIRDS = 0.66,
		        FOUR = 4;

		public static double sqr(double x) {
			return x * x;
		}

		public static double max3(double x, double y, double z) {
			return x < y ? (y < z ? z : y) : (x < z ? z : x);
		}

		/**
		 * Minimize a function along a search direction. This code is a Java
		 * translation of the function <code>MCSRCH</code> from
		 * <code>lbfgs.f</code> , which in turn is a slight modification of the
		 * subroutine <code>CSRCH</code> of More' and Thuente. The changes are
		 * to allow reverse communication, and do not affect the performance of
		 * the routine. This function, in turn, calls <code>mcstep</code>.
		 * <p>
		 * 
		 * The Java translation was effected mostly mechanically, with some
		 * manual clean-up; in particular, array indices start at 0 instead of
		 * 1. Most of the comments from the Fortran code have been pasted in
		 * here as well.
		 * <p>
		 * 
		 * The purpose of <code>mcsrch</code> is to find a step which satisfies
		 * a sufficient decrease condition and a curvature condition.
		 * <p>
		 * 
		 * At each stage this function updates an interval of uncertainty with
		 * endpoints <code>stx</code> and <code>sty</code>. The interval of
		 * uncertainty is initially chosen so that it contains a minimizer of
		 * the modified function
		 * 
		 * <pre>
		 *      f(x+stp*s) - f(x) - ftol*stp*(gradf(x)'s).
		 * </pre>
		 * 
		 * If a step is obtained for which the modified function has a
		 * nonpositive function value and nonnegative derivative, then the
		 * interval of uncertainty is chosen so that it contains a minimizer of
		 * <code>f(x+stp*s)</code>.
		 * <p>
		 * 
		 * The algorithm is designed to find a step which satisfies the
		 * sufficient decrease condition
		 * 
		 * <pre>
		 *       f(x+stp*s) &lt;= f(X) + ftol*stp*(gradf(x)'s),
		 * </pre>
		 * 
		 * and the curvature condition
		 * 
		 * <pre>
		 *       abs(gradf(x+stp*s)'s)) &lt;= gtol*abs(gradf(x)'s).
		 * </pre>
		 * 
		 * If <code>ftol</code> is less than <code>gtol</code> and if, for
		 * example, the function is bounded below, then there is always a step
		 * which satisfies both conditions. If no step can be found which
		 * satisfies both conditions, then the algorithm usually stops when
		 * rounding errors prevent further progress. In this case
		 * <code>stp</code> only satisfies the sufficient decrease condition.
		 * <p>
		 * 
		 * @author Original Fortran version by Jorge J. More' and David J.
		 *         Thuente as part of the Minpack project, June 1983, Argonne
		 *         National Laboratory. Java translation by Robert Dodier,
		 *         August 1997.
		 * 
		 * @param n
		 *            The number of variables.
		 * 
		 * @param x
		 *            On entry this contains the base point for the line search.
		 *            On exit it contains <code>x + stp*s</code>.
		 * 
		 * @param f
		 *            On entry this contains the value of the objective function
		 *            at <code>x</code>. On exit it contains the value of the
		 *            objective function at <code>x + stp*s</code>.
		 * 
		 * @param g
		 *            On entry this contains the gradient of the objective
		 *            function at <code>x</code>. On exit it contains the
		 *            gradient at <code>x + stp*s</code>.
		 * 
		 * @param s
		 *            The search direction.
		 * 
		 * @param stp
		 *            On entry this contains an initial estimate of a
		 *            satifactory step length. On exit <code>stp</code> contains
		 *            the final estimate.
		 * 
		 * @param ftol
		 *            Tolerance for the sufficient decrease condition.
		 * 
		 * @param xtol
		 *            Termination occurs when the relative width of the interval
		 *            of uncertainty is at most <code>xtol</code>.
		 * 
		 * @param maxfev
		 *            Termination occurs when the number of evaluations of the
		 *            objective function is at least <code>maxfev</code> by the
		 *            end of an iteration.
		 * 
		 * @param info
		 *            This is an output variable, which can have these values:
		 *            <ul>
		 *            <li><code>info = 0</code> Improper input parameters. <li>
		 *            <code>info = -1</code> A return is made to compute the
		 *            function and gradient. <li><code>info = 1</code> The
		 *            sufficient decrease condition and the directional
		 *            derivative condition hold. <li><code>info = 2</code>
		 *            Relative width of the interval of uncertainty is at most
		 *            <code>xtol</code>. <li> <code>info = 3</code> Number of
		 *            function evaluations has reached <code>maxfev</code>. <li>
		 *            <code>info = 4</code> The step is at the lower bound
		 *            <code>stpmin</code>. <li><code>info = 5</code> The step is
		 *            at the upper bound <code>stpmax</code>. <li><code>info = 6
		 *            </code> Rounding errors prevent further progress. There
		 *            may not be a step which satisfies the sufficient decrease
		 *            and curvature conditions. Tolerances may be too small.
		 *            </ul>
		 * 
		 * @param nfev
		 *            On exit, this is set to the number of function
		 *            evaluations.
		 * 
		 * @param wa
		 *            Temporary storage array, of length <code>n</code>.
		 */

		public static void mcsrch(final int n, double[] x, double f,
		        double[] g, double[] s, int is0, double[] stp, double ftol,
		        double xtol, int maxfev, int[] info, int[] nfev, double[] wa,
		        int[] iprint) {

			if (info[0] != -1) {
				infoc[0] = 1;
				if (n <= 0 || stp[0] <= 0 || ftol < 0 || LBFGS.gtol < 0
				        || xtol < 0 || LBFGS.stpmin < 0
				        || LBFGS.stpmax < LBFGS.stpmin || maxfev <= 0)
					return;

				// Compute the initial gradient in the search direction
				// and check that s is a descent direction.
				dginit = 0;
				for (j = 1; j <= n; j += 1) {
					dginit += g[j - 1] * s[is0 + j - 1];
					if (Double.isNaN(dginit))
						System.err.println("NaN " + g[j - 1] + " "
						        + s[is0 + j - 1]);
				}
				if (dginit >= 0) {
					System.out
					        .println("The search direction is not a descent direction.");
					return;
				}

				brackt[0] = false;
				stage1 = true;
				nfev[0] = 0;
				finit = f;
				dgtest = ftol * dginit;
				width = LBFGS.stpmax - LBFGS.stpmin;
				width1 = width / ONE_HALF;

				for (j = 1; j <= n; j += 1) {
					wa[j - 1] = x[j - 1];
				}

				// The variables stx, fx, dgx contain the values of the step,
				// function, and directional derivative at the best step.
				// The variables sty, fy, dgy contain the value of the step,
				// function, and derivative at the other endpoint of
				// the interval of uncertainty.
				// The variables stp, f, dg contain the values of the step,
				// function, and derivative at the current step.
				stx[0] = 0;
				fx[0] = finit;
				dgx[0] = dginit;
				sty[0] = 0;
				fy[0] = finit;
				dgy[0] = dginit;

				if (iprint[0] > 0)
					System.err.println("new line search f=" + finit + " dg="
					        + dginit + " stp=" + stp[0]);
			}

			while (true) {
				if (info[0] != -1) {
					// Set the minimum and maximum steps to correspond
					// to the present interval of uncertainty.
					if (brackt[0]) {
						stmin = Math.min(stx[0], sty[0]);
						stmax = Math.max(stx[0], sty[0]);
					} else {
						stmin = stx[0];
						stmax = stp[0] + FOUR * (stp[0] - stx[0]);
					}

					// Force the step to be within the bounds stpmax and stpmin.
					setSTP(stp, Math.max(stp[0], LBFGS.stpmin));
					setSTP(stp, Math.min(stp[0], LBFGS.stpmax));

					// If an unusual termination is to occur then let
					// stp be the lowest point obtained so far.
					if ((brackt[0] && (stp[0] <= stmin || stp[0] >= stmax || stmax
					        - stmin <= xtol * stmax))
					        || nfev[0] >= maxfev - 1 || infoc[0] == 0)
						setSTP(stp, stx[0]);

					// Evaluate the function and gradient at stp
					// and compute the directional derivative.
					// We return to main program to obtain F and G.
					for (j = 1; j <= n; j += 1) {
						x[j - 1] = wa[j - 1] + stp[0] * s[is0 + j - 1];
						if (Math.abs(x[j - 1]) > 40) {
							System.err.println("big wt " + j + " " + stp[0]
							        + " " + s[is0 + j - 1]);
						}
					}

					info[0] = -1;
					return;
				}

				info[0] = 0;
				nfev[0] = nfev[0] + 1;
				dg = 0;

				for (j = 1; j <= n; j += 1) {
					dg = dg + g[j - 1] * s[is0 + j - 1];
				}

				ftest1 = finit + stp[0] * dgtest;

				// Test for convergence.
				if ((brackt[0] && (stp[0] <= stmin || stp[0] >= stmax))
				        || infoc[0] == 0)
					// Rounding errors prevent further
					// progress. There may not be a step which satisfies the
					// sufficient decrease and curvature conditions. Tolerances
					// may
					// be too small.
					info[0] = 6;

				if (stp[0] == LBFGS.stpmax && f <= ftest1 && dg <= dgtest)
					// The step is at the upper bound <code>stpmax</code>.
					info[0] = 5;

				if (stp[0] == LBFGS.stpmin && (f > ftest1 || dg >= dgtest))
					// The step is at the lower bound <code>stpmin</code>.
					info[0] = 4;

				if (nfev[0] >= maxfev)
					// Number of function evaluations has reached
					// <code>maxfev</code>
					info[0] = 3;

				if (brackt[0] && stmax - stmin <= xtol * stmax)
					// Relative width of the interval of uncertainty is at most
					// <code>xtol</code>
					info[0] = 2;

				if (f <= ftest1 && Math.abs(dg) <= LBFGS.gtol * (-dginit))
					// The sufficient decrease condition and the directional
					// derivative condition hold.
					info[0] = 1;

				// Check for termination.
				if (info[0] != 0) {
					info[0] = 1;
					return;
				}

				// In the first stage we seek a step for which the modified
				// function has a nonpositive value and nonnegative derivative.
				if (stage1 && f <= ftest1
				        && dg >= Math.min(ftol, LBFGS.gtol) * dginit)
					stage1 = false;

				// A modified function is used to predict the step only if
				// we have not obtained a step for which the modified
				// function has a nonpositive function value and nonnegative
				// derivative, and if a lower function value has been
				// obtained but the decrease is not sufficient.
				if (stage1 && f <= fx[0] && f > ftest1) {
					// Define the modified function and derivative values.
					fm = f - stp[0] * dgtest;
					fxm[0] = fx[0] - stx[0] * dgtest;
					fym[0] = fy[0] - sty[0] * dgtest;
					dgm = dg - dgtest;
					dgxm[0] = dgx[0] - dgtest;
					dgym[0] = dgy[0] - dgtest;

					// Call cstep to update the interval of uncertainty
					// and to compute the new step.
					mcstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm,
					        brackt, stmin, stmax, infoc, iprint);

					// Reset the function and gradient values for f.
					fx[0] = fxm[0] + stx[0] * dgtest;
					fy[0] = fym[0] + sty[0] * dgtest;
					dgx[0] = dgxm[0] + dgtest;
					dgy[0] = dgym[0] + dgtest;
				} else {
					// Call mcstep to update the interval of uncertainty
					// and to compute the new step.
					mcstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt,
					        stmin, stmax, infoc, iprint);
				}

				if (iprint[0] > 0)
					System.err.println(" msrch internal f=" + f + " dg=" + dg
					        + " stx=" + stx[0] + " sty=" + sty[0] + " brackt="
					        + brackt[0] + " stp=" + stp[0]);

				// Force a sufficient decrease in the size of the
				// interval of uncertainty.
				if (brackt[0]) {
					if (Math.abs(sty[0] - stx[0]) >= TWO_THIRDS * width1)
						setSTP(stp, (sty[0] + stx[0]) / 2.0);
					width1 = width;
					width = Math.abs(sty[0] - stx[0]);
				}
			}
		}

		/**
		 * The purpose of this function is to compute a safeguarded step for a
		 * linesearch and to update an interval of uncertainty for a minimizer
		 * of the function.
		 * <p>
		 * 
		 * The parameter <code>stx</code> contains the step with the least
		 * function value. The parameter <code>stp</code> contains the current
		 * step. It is assumed that the derivative at <code>stx</code> is
		 * negative in the direction of the step. If <code>brackt[0]</code> is
		 * <code>true</code> when <code>mcstep</code> returns then a minimizer
		 * has been bracketed in an interval of uncertainty with endpoints
		 * <code>stx</code> and <code>sty</code>.
		 * <p>
		 * 
		 * Variables that must be modified by <code>mcstep</code> are
		 * implemented as 1-element arrays.
		 * 
		 * @param stx1
		 *            Step at the best step obtained so far. This variable is
		 *            modified by <code>mcstep</code>.
		 * @param fx1
		 *            Function value at the best step obtained so far. This
		 *            variable is modified by <code>mcstep</code>.
		 * @param dx
		 *            Derivative at the best step obtained so far. The
		 *            derivative must be negative in the direction of the step,
		 *            that is, <code>dx</code> and <code>stp-stx</code> must
		 *            have opposite signs. This variable is modified by
		 *            <code>mcstep</code>.
		 * 
		 * @param sty1
		 *            Step at the other endpoint of the interval of uncertainty.
		 *            This variable is modified by <code>mcstep</code>.
		 * @param fy1
		 *            Function value at the other endpoint of the interval of
		 *            uncertainty. This variable is modified by
		 *            <code>mcstep</code>.
		 * @param dy
		 *            Derivative at the other endpoint of the interval of
		 *            uncertainty. This variable is modified by
		 *            <code>mcstep</code>.
		 * 
		 * @param stp
		 *            Step at the current step. If <code>brackt</code> is set
		 *            then on input <code>stp</code> must be between
		 *            <code>stx</code> and <code>sty</code>. On output
		 *            <code>stp</code> is set to the new step.
		 * @param fp
		 *            Function value at the current step.
		 * @param dp
		 *            Derivative at the current step.
		 * 
		 * @param brackt1
		 *            Tells whether a minimizer has been bracketed. If the
		 *            minimizer has not been bracketed, then on input this
		 *            variable must be set <code>false</code>. If the minimizer
		 *            has been bracketed, then on output this variable is
		 *            <code>true</code>.
		 * 
		 * @param stpmin
		 *            Lower bound for the step.
		 * @param stpmax
		 *            Upper bound for the step.
		 * 
		 * @param info
		 *            On return from <code>mcstep</code>, this is set as
		 *            follows: If <code>info</code> is 1, 2, 3, or 4, then the
		 *            step has been computed successfully. Otherwise
		 *            <code>info</code> = 0, and this indicates improper input
		 *            parameters.
		 * 
		 * @author Jorge J. More, David J. Thuente: original Fortran version, as
		 *         part of Minpack project. Argonne Nat'l Laboratory, June 1983.
		 *         Robert Dodier: Java translation, August 1997.
		 */
		public static void mcstep(double[] stx1, double[] fx1, double[] dx,
		        double[] sty1, double[] fy1, double[] dy, double[] stp,
		        double fp, double dp, boolean[] brackt1, double stpmin,
		        double stpmax, int[] info, int[] iprint) {
			boolean bound;
			double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;

			info[0] = 0;

			if ((brackt1[0] && (stp[0] <= Math.min(stx1[0], sty1[0]) || stp[0] >= Math
			        .max(stx1[0], sty1[0])))
			        || dx[0] * (stp[0] - stx1[0]) >= 0.0 || stpmax < stpmin) {
				if (iprint[0] > 0)
					System.err.println("mcstep=0 " + brackt1[0] + " " + stp[0]
					        + " " + stx1[0] + " " + sty1[0] + " " + dx[0] + " "
					        + stpmax + " " + stpmin);
				return;
			}

			// Determine if the derivatives have opposite sign.
			sgnd = dp * (dx[0] / Math.abs(dx[0]));

			if (fp > fx1[0]) {
				// First case. A higher function value.
				// The minimum is bracketed. If the cubic step is closer
				// to stx than the quadratic step, the cubic step is taken,
				// else the average of the cubic and quadratic steps is taken.

				info[0] = 1;
				bound = true;
				theta = 3 * (fx1[0] - fp) / (stp[0] - stx1[0]) + dx[0] + dp;
				s = max3(Math.abs(theta), Math.abs(dx[0]), Math.abs(dp));
				gamma = s * Math.sqrt(sqr(theta / s) - (dx[0] / s) * (dp / s));
				if (stp[0] < stx1[0])
					gamma = -gamma;
				p = (gamma - dx[0]) + theta;
				q = ((gamma - dx[0]) + gamma) + dp;
				r = p / q;
				stpc = stx1[0] + r * (stp[0] - stx1[0]);
				stpq = stx1[0]
				        + ((dx[0] / ((fx1[0] - fp) / (stp[0] - stx1[0]) + dx[0])) / 2)
				        * (stp[0] - stx1[0]);
				if (Math.abs(stpc - stx1[0]) < Math.abs(stpq - stx1[0])) {
					stpf = stpc;
				} else {
					stpf = stpc + (stpq - stpc) / 2;
				}
				brackt1[0] = true;
			} else if (sgnd < 0.0) {
				// Second case. A lower function value and derivatives of
				// opposite sign. The minimum is bracketed. If the cubic
				// step is closer to stx than the quadratic (secant) step,
				// the cubic step is taken, else the quadratic step is taken.

				info[0] = 2;
				bound = false;
				theta = 3 * (fx1[0] - fp) / (stp[0] - stx1[0]) + dx[0] + dp;
				s = max3(Math.abs(theta), Math.abs(dx[0]), Math.abs(dp));
				gamma = s * Math.sqrt(sqr(theta / s) - (dx[0] / s) * (dp / s));
				if (stp[0] > stx1[0])
					gamma = -gamma;
				p = (gamma - dp) + theta;
				q = ((gamma - dp) + gamma) + dx[0];
				r = p / q;
				stpc = stp[0] + r * (stx1[0] - stp[0]);
				stpq = stp[0] + (dp / (dp - dx[0])) * (stx1[0] - stp[0]);
				if (Math.abs(stpc - stp[0]) > Math.abs(stpq - stp[0])) {
					stpf = stpc;
				} else {
					stpf = stpq;
				}
				brackt1[0] = true;
			} else if (Math.abs(dp) < Math.abs(dx[0])) {
				// Third case. A lower function value, derivatives of the
				// same sign, and the magnitude of the derivative decreases.
				// The cubic step is only used if the cubic tends to infinity
				// in the direction of the step or if the minimum of the cubic
				// is beyond stp. Otherwise the cubic step is defined to be
				// either stpmin or stpmax. The quadratic (secant) step is also
				// computed and if the minimum is bracketed then the the step
				// closest to stx is taken, else the step farthest away is
				// taken.

				info[0] = 3;
				bound = true;
				theta = 3 * (fx1[0] - fp) / (stp[0] - stx1[0]) + dx[0] + dp;
				s = max3(Math.abs(theta), Math.abs(dx[0]), Math.abs(dp));
				gamma = s
				        * Math.sqrt(Math.max(0, sqr(theta / s) - (dx[0] / s)
				                * (dp / s)));
				if (stp[0] > stx1[0])
					gamma = -gamma;
				p = (gamma - dp) + theta;
				q = (gamma + (dx[0] - dp)) + gamma;
				r = p / q;
				if (r < 0.0 && gamma != 0.0) {
					stpc = stp[0] + r * (stx1[0] - stp[0]);
				} else if (stp[0] > stx1[0]) {
					stpc = stpmax;
				} else {
					stpc = stpmin;
				}
				stpq = stp[0] + (dp / (dp - dx[0])) * (stx1[0] - stp[0]);
				if (brackt1[0]) {
					if (Math.abs(stp[0] - stpc) < Math.abs(stp[0] - stpq)) {
						stpf = stpc;
					} else {
						stpf = stpq;
					}
				} else {
					if (Math.abs(stp[0] - stpc) > Math.abs(stp[0] - stpq)) {
						stpf = stpc;
					} else {
						stpf = stpq;
					}
				}
			} else {
				// Fourth case. A lower function value, derivatives of the
				// same sign, and the magnitude of the derivative does
				// not decrease. If the minimum is not bracketed, the step
				// is either stpmin or stpmax, else the cubic step is taken.

				info[0] = 4;
				bound = false;
				if (brackt1[0]) {
					theta = 3 * (fp - fy1[0]) / (sty1[0] - stp[0]) + dy[0] + dp;
					s = max3(Math.abs(theta), Math.abs(dy[0]), Math.abs(dp));
					gamma = s
					        * Math.sqrt(sqr(theta / s) - (dy[0] / s) * (dp / s));
					if (stp[0] > sty1[0])
						gamma = -gamma;
					p = (gamma - dp) + theta;
					q = ((gamma - dp) + gamma) + dy[0];
					r = p / q;
					stpc = stp[0] + r * (sty1[0] - stp[0]);
					stpf = stpc;
				} else if (stp[0] > stx1[0]) {
					stpf = stpmax;
				} else {
					stpf = stpmin;
				}
			}

			// Update the interval of uncertainty. This update does not
			// depend on the new step or the case analysis above.

			if (fp > fx1[0]) {
				sty1[0] = stp[0];
				fy1[0] = fp;
				dy[0] = dp;
			} else {
				if (sgnd < 0.0) {
					sty1[0] = stx1[0];
					fy1[0] = fx1[0];
					dy[0] = dx[0];
				}
				stx1[0] = stp[0];
				fx1[0] = fp;
				dx[0] = dp;
			}

			// Compute the new step and safeguard it.

			stpf = Math.min(stpmax, stpf);
			stpf = Math.max(stpmin, stpf);
			setSTP(stp, stpf);

			if (brackt1[0] && bound) {
				double possibleStep = stx1[0] + TWO_THIRDS
				        * (sty1[0] - stx1[0]);
				if (sty1[0] > stx1[0]) {
					setSTP(stp, Math.min(possibleStep, stp[0]));
				} else {
					setSTP(stp, Math.max(possibleStep, stp[0]));
				}
			}
			// System.err.println("mcsetp stp => " + stp[0] + " info=" +
			// info[0]);

			return;
		}

		public static void setSTP(double[] stp, double value) {
			if (value < 0)// || value > 10)
				throw new IllegalArgumentException(value + "");
			if (value > 1000) {
				stp[0] = 1000;
				System.err.println("Reducing step from " + value + " to 1000");
			} else {
				stp[0] = value;
			}
		}

	}

	public static void main(String args[]) {
		int ndim = 2000, msave = 7;
		int nwork = ndim * (2 * msave + 1) + 2 * msave;
		double[] x, g, diag;
		x = new double[ndim];
		g = new double[ndim];
		diag = new double[ndim];
		w = new double[nwork];

		double f, eps, xtol, t1, t2;
		int iprint[], iflag[] = new int[1], icall, n, m, j;
		iprint = new int[2];
		boolean diagco;

		n = 100;
		m = 5;
		iprint[1 - 1] = 1;
		iprint[2 - 1] = 0;
		diagco = false;
		eps = 1.0e-5;
		xtol = 1.0e-16;
		icall = 0;
		iflag[0] = 0;

		for (j = 1; j <= n; j += 2) {
			x[j - 1] = -1.2e0;
			x[j + 1 - 1] = 1.e0;
		}

		do {
			f = 0;
			for (j = 1; j <= n; j += 2) {
				t1 = 1.e0 - x[j - 1];
				t2 = 1.e1 * (x[j + 1 - 1] - x[j - 1] * x[j - 1]);
				g[j + 1 - 1] = 2.e1 * t2;
				g[j - 1] = -2.e0 * (x[j - 1] * g[j + 1 - 1] + t1);
				f = f + t1 * t1 + t2 * t2;
			}

			try {
				LBFGS.lbfgs(n, m, x, f, g, diagco, diag, iprint, eps, xtol,
				        iflag);
			} catch (Exception e) {
				e.printStackTrace();
				return;
			}
			icall++;
		} while (iflag[0] != 0 && icall <= 200);
	}
}