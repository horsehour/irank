package com.horsehour.math.opt.ga;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;

import com.horsehour.math.function.TransferFunction;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since Sep. 10 2011
 */
public class GeneticAlgorithm {
	private int numGeneration;
	private Population pop;
	private File outputFile;

	public GeneticAlgorithm() {}

	/**
	 * @param popsize
	 *            size of population
	 * @param numGeneration
	 *            number of generations to evolve
	 * @param cr
	 *            cross over rate
	 * @param lBound
	 *            lower bound
	 * @param uBound
	 *            upper bound
	 * @param granual
	 *            grannual to search the feasible region
	 * @param fitFunc
	 *            fitness function
	 */
	public GeneticAlgorithm(int popsize, int numGeneration, float cr, double lBound, double uBound,
	        double granual, TransferFunction fitFunc) {
		this.numGeneration = numGeneration;
		this.pop = new Population(popsize, cr, fitFunc);
		this.pop.setFeasibleRegion(lBound, uBound, granual);
		this.pop.initPopulation();
	}

	/**
	 * start up the evolution process
	 */
	public void evolve(){
		for (int i = numGeneration; i > 0; i--) {
			record();
			pop.evolve();
		}
	}

	/**
	 * record the evolution process
	 */
	private void record(){
		StringBuffer sb = new StringBuffer();
		sb.append("# Number of Generation:\t" + pop.getAge() + "\r\n");
		int sz = pop.getSize();
		Individual idv = new Individual();
		double sum = 0;
		for (int i = 0; i < sz; i++) {
			idv = pop.getMember(i);
			double x = pop.decode(idv);
			double y = pop.evalFitness(x);
			sb.append("  " + (i + 1) + "-" + x + "\t" + y + "\r\n");
			sum += y;
		}

		sb.append("# Average Objective Value:\t" + sum / sz + "\r\n");
		try {
			FileUtils.write(outputFile, sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void setOutputFile(File destFile){
		this.outputFile = destFile;
	}

	public void setNumGeneration(int numGeneration){
		this.numGeneration = numGeneration;
	}

	public int getNumGeneration(){
		return numGeneration;
	}

	public Population getPopulation(){
		return pop;
	}
}