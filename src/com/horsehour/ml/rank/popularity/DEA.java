package com.horsehour.ml.rank.popularity;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;

/**
 * Class that performs Data Envelopment Analysis. It requires the installation of
 * lpsolve 5.5.2.0. http://lpsolve.sourceforge.net/5.5/
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class DEA {
	/**
	 * The DEA Constraint is an internal class which stores the the main body of
	 * the constraint, the sign and the right value.
	 */
	private class DEAConstraint {
		/**
		 * The body of the constraint; the left part of the constraint.
		 */
		public final double[] contraintBody;

		/**
		 * The sign of the constraint: LE (1) for <=, EQ (3) for =, GE (2) for >=
		 */
		public final int sign;

		/**
		 * The value of the constraint; the right part of the constraint.
		 */
		public final double value;

		/**
		 * The constructor of DEA Constraint.
		 * 
		 * @param constraintBody    The array with the parameters of the constrain
		 * @param sign              The sign of the constrain
		 * @param value             The right part value of the constrain
		 */
		public DEAConstraint(double[] constraintBody, int sign, double value) {
			this.contraintBody = constraintBody;
			this.sign = sign;
			this.value = value;
		}
	}

	/**
	 * Checks whether the status corresponds to a valid solution
	 * 
	 * @param status    The status id returned by lpsolve
	 * @return          Boolean which indicates if the solution is valid
	 */
	private boolean isSolutionValid(int status){
		return (status == 0) || (status == 1) || (status == 11) || (status == 12);
	}

	/**
	 * Estimates the efficiency of the records by running DEA
	 * 
	 * @param records   Map with the DeaRecords
	 * @return          Map with the scores of the records
	 * @throws LpSolveException 
	 */
	public Map<String, Double> estimateEfficiency(Map<String, DEARecord> records)
	        throws LpSolveException{
		// Important note. All the double[] arrays that are passed to the
		// lpsolve lib MUST start from 1. Do not use the 0 index because it is
		// discarded.
		Map<String, Double> evaluatedResults = new LinkedHashMap<>();

		List<DEAConstraint> constraints = new ArrayList<>();

		// initialize the constraints list
		Integer totalColumns = null;
		boolean hasInput = false;
		for (Map.Entry<String, DEARecord> entry : records.entrySet()) {
			DEARecord currentRecord = entry.getValue();
			int currentColumns = currentRecord.getInput().length; // add the size of input array
			boolean currentHasInput = (currentColumns > 0); // check if the input is defined

			currentColumns += currentRecord.getOutput().length; // add the size of output array

			if (totalColumns == null) { // if totalColumns is not set, then set them
				// the totalColumns is the sum of input and output columns
				totalColumns = currentColumns;
				hasInput = currentHasInput;
			} else { // if totalColumns is initialized, validate that the record
				     // has exactly this amount of columns
				if (totalColumns != currentColumns) {
					throw new IllegalArgumentException(
					        "The input and output columns do not match in all records.");
				}
				if (hasInput != currentHasInput) {
					throw new IllegalArgumentException(
					        "The input should be used in all records or in none.");
				}
			}

			// We have two cases. Either an input is defined for the records or
			// not. The mathematical model is formulated differently depending
			// the case
			if (hasInput == false) {
				// if no input then change the way that the linear problem
				// formulates
				constraints.add(new DEAConstraint(currentRecord.getOutput(), LpSolve.LE, 1.0));
			} else {
				// create a double[] with size both of the input and output
				double[] currentConstraintBody = new double[totalColumns + 1];

				// set the values of output first on the new array
				double[] conOutput = currentRecord.getOutput();
				for (int i = 0; i < conOutput.length; ++i) {
					// IMPORTANT! Start from 1!
					currentConstraintBody[i + 1] = conOutput[i];
				}

				// now set the input by negatiting the values
				double[] conInput = currentRecord.getInput();
				for (int i = 0; i < conInput.length; ++i) {
					// IMPORTANT! Start from 1!
					currentConstraintBody[conOutput.length + i + 1] = -conInput[i];
				}
				conOutput = null;
				conInput = null;

				// add the constrain on the list
				constraints.add(new DEAConstraint(currentConstraintBody, LpSolve.LE, 0.0));
			}
		}

		// evaluate the ranking of the records. In case we are not interested in
		// the scores of all the records we can extend this method to evaluate
		// the ranking of only the records of interest and thus speed up
		// dramatically
		// the speed.
		LpSolve solver;
		for (Map.Entry<String, DEARecord> entry : records.entrySet()) {
			String currentRecordId = entry.getKey();
			DEARecord currentRecord = entry.getValue();

			solver = LpSolve.makeLp(0, totalColumns);
			solver.setVerbose(LpSolve.NEUTRAL); // set verbose level
			solver.setMaxim(); // set the problem to maximization

			// add all the Constraints in solver
			for (DEAConstraint constraint : constraints) {
				solver.addConstraint(constraint.contraintBody, constraint.sign, constraint.value);
			}

			if (hasInput == false) {
				// set the Objection function equal to the output of the record
				solver.setObjFn(currentRecord.getOutput());
			} else {
				// create a double[] with size both of the input and output
				double[] objectiveFunction = new double[totalColumns + 1];
				double[] denominatorConstraintBody = new double[totalColumns + 1];

				// set the values of output first on the new array
				double[] conOutput = currentRecord.getOutput();
				for (int i = 0; i < conOutput.length; ++i) {
					// IMPORTANT! Start from 1!
					// set the output to the objective function
					objectiveFunction[i + 1] = conOutput[i];
					// set zero to the constraint
					denominatorConstraintBody[i + 1] = 0.0;
				}

				// set the values of input first on the new array
				double[] conInput = currentRecord.getInput();
				for (int i = 0; i < conInput.length; ++i) {
					// IMPORTANT! Start from 1!
					//set zeros on objective function for input
					denominatorConstraintBody[conOutput.length + i + 1] = 0.0;
					//set the input to the constraint
					denominatorConstraintBody[conOutput.length + i + 1] = conInput[i];
				}
				conInput = null;
				conOutput = null;

				// set the denominator equal to 1
				solver.addConstraint(denominatorConstraintBody, LpSolve.EQ, 1.0);

				// set objective function
				solver.setObjFn(objectiveFunction);
			}

			int status = solver.solve();
			if (isSolutionValid(status) == false) {
				solver.setScaling(LpSolve.SCALE_NONE);//turn off automatic scaling
				status = solver.solve();
				if (isSolutionValid(status) == false) {
					System.err.println("Error! " + solver.getStatustext(status));
					solver.printLp();
				}
			}

			evaluatedResults.put(currentRecordId, solver.getObjective());
			solver.deleteLp();// delete problem and free the memory
		}

		return evaluatedResults;
	}
}
