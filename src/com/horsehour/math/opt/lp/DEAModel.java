package com.horsehour.math.opt.lp;

import java.util.Arrays;

import org.opensourcedea.dea.DEAProblem;
import org.opensourcedea.dea.ModelType;
import org.opensourcedea.dea.SolverReturnStatus;
import org.opensourcedea.dea.VariableOrientation;

public class DEAModel {

	private DEAProblem problem = null;

	private DEAProblem designProblem(double[][] data) {
		int nDMU = data.length, nVar = data[0].length + 1;

		DEAProblem problem = new DEAProblem(nDMU, nVar);

		VariableOrientation[] varOrientation = new VariableOrientation[nVar];

		for (int idx = 0; idx < nVar - 1; idx++)
			varOrientation[idx] = VariableOrientation.OUTPUT;// for the second
															 // model: INPUT

		varOrientation[nVar - 1] = VariableOrientation.INPUT;// for the second
															 // model: OUTPUT

		double[][] dataMatrix = new double[nDMU][nVar];

		for (int i = 0; i < nDMU; i++) {
			for (int j = 0; j < nVar - 1; j++)
				dataMatrix[i][j] = data[i][j];
			dataMatrix[i][nVar - 1] = 1;// for the second model: =y
		}

		problem.setDataMatrix(dataMatrix);
		problem.setVariableOrientations(varOrientation);
		problem.setModelType(ModelType.CCR_O);

		return problem;
	}

	public double[] solveProblem(double[][] data) {
		designProblem(data);

		double[][] efficientWeight = null;
		double[] weight = null;
		try {
			problem.solve();
			SolverReturnStatus status = problem.getOptimisationStatus();

			if (SolverReturnStatus.OPTIMAL_SOLUTION_FOUND.equals(status)) {

				efficientWeight = problem.getWeight();

				int sz = problem.getNumberOfVariables();
				for (int id = 0; id < problem.getNumberOfDMUs(); id++)
					weight = Arrays.copyOf(efficientWeight[id], sz - 1);
			}

		} catch (Exception e) {
			return null;
		}
		return weight;
	}
}
