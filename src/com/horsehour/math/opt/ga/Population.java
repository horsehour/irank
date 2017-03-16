package com.horsehour.math.opt.ga;

import java.util.BitSet;

import com.horsehour.math.function.TransferFunction;
import com.horsehour.util.MathLib;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since Dec. 18 2012
 **/
public class Population {
	private Individual[] members;
	private Individual[] elitMembers;

	private Double[] fit;
	private final TransferFunction fitFunc;

	private final int size;
	private int age;
	private int cl;// cross length
	private int capacity;
	private final float cr;

	private double lBound;
	private double uBound;
	private double granual;// 问题空间切分粒度,如10E-3

	/**
	 * @param sz
	 *            number of members in the population
	 * @param capacity
	 *            capacity of each member(individual,gene)
	 * @param cr
	 *            cross over rate
	 */
	public Population(int sz, float cr, TransferFunction fitFunc) {
		this.size = sz;
		this.cr = cr;
		this.fitFunc = fitFunc;
	}

	/**
	 * set feasible region
	 * 
	 * @param lBound
	 *            lower bound
	 * @param uBound
	 *            upper bound
	 * @param granual
	 *            grannual to search the feasible region
	 */
	public void setFeasibleRegion(double lBound, double uBound, double granual){
		this.lBound = lBound;
		this.uBound = uBound;
		this.granual = granual;
		this.capacity = estimateLeastCapacity();
	}

	/**
	 * estimate the least capacity of each gene according to the bounds and
	 * granual
	 * 
	 * @return length of gene
	 */
	private int estimateLeastCapacity(){
		double range = uBound - lBound;
		double temp = Math.log10(range / granual) / Math.log10(2);
		return new Double(Math.floor(temp)).intValue() + 1;
	}

	/**
	 * initialize the population
	 */
	public void initPopulation(){
		this.cl = (int) (capacity * cr);
		this.age = 1;// 第一代
		members = new Individual[size];

		for (int i = 0; i < size; i++) {
			double rnd = MathLib.Rand.uniform(lBound, uBound);
			members[i] = new Individual(capacity, encode(rnd));
		}
	}

	public void evolve(){
		fit = getFitness();
		selectRouletteWheel();

		int numPair = 0;
		if (size % 2 == 0)
			numPair = size / 2;
		else {
			numPair = (size - 1) / 2;
			inherit(members[size - 1], elitMembers[size - 1]);
		}

		for (int i = 0; i < numPair; i++) {
			crossover(elitMembers[2 * i], elitMembers[2 * i + 1]);
			inherit(members[2 * i], elitMembers[2 * i]);
			inherit(members[2 * i + 1], elitMembers[2 * i + 1]);
		}
		age++;// 产生了新的一代
	}

	/**
	 * @return compute the fitness(0 - 1) of each individual
	 */
	private Double[] getFitness(){
		Double[] fitness = new Double[size];
		double sum = 0;
		for (int i = 0; i < size; i++) {
			fitness[i] = evalFitness(members[i]);
			sum += fitness[i];
		}

		if (sum == 0)
			return fitness;
		return MathLib.Matrix.multiply(fitness, 1.0 / sum);
	}

	/**
	 * @param val
	 * @return fitness value based on val
	 */
	public double evalFitness(double val){
		return fitFunc.calc(val);
	}

	public double evalFitness(Individual indv){
		return evalFitness(decode(indv));
	}

	/**
	 * @return cumulative value of fitness
	 */
	private double[] cumulativeFitness(){
		double[] cf = new double[size];
		cf[0] = fit[0];
		for (int i = 1; i < size; i++)
			cf[i] = fit[i] + cf[i - 1];
		return cf;
	}

	/**
	 * select the fittest individual in a roulette wheel manner
	 */
	private void selectRouletteWheel(){
		double[] cf = cumulativeFitness();
		elitMembers = new Individual[size];

		for (int i = 0; i < size; i++) {
			elitMembers[i] = new Individual(capacity);
			double rnd = Math.random();
			for (int j = 0; j < size; j++)
				if (rnd <= cf[j]) {
					inherit(elitMembers[i], members[j]);
					break;
				}
		}
	}

	/**
	 * <p>
	 * cross over (one-point) two individual
	 * <p>
	 * 1)select a pivot point
	 * <p>
	 * 2)cross over with given rate
	 * 
	 * @param idv1
	 * @param idv2
	 */
	private void crossover(Individual idv1, Individual idv2){
		int tail = capacity - 1;
		int pivot = MathLib.Rand.sample(0, capacity);

		int rightEnd = pivot + cl;
		int leftLen = rightEnd - tail;
		if (leftLen <= 0) {
			idv1.onepoint(idv2, pivot, rightEnd);
			return;
		}

		idv1.onepoint(idv2, pivot, tail);
		idv1.onepoint(idv2, 0, leftLen - 1);
	}

	/**
	 * @param indv
	 * @return decode chromosome in bin to dec
	 */
	public double decode(Individual indv){
		BitSet chromosome = indv.getChromosome();
		double code = 0;
		for (int i = 0; i < capacity; i++) {
			if (chromosome.get(i))
				code += Math.pow(2, capacity - 1 - i);
		}
		code = lBound + code * granual;
		return code;
	}

	/**
	 * @param val
	 * @return encode val in dec to bin
	 */
	public BitSet encode(double val){
		int m = (int) ((val - lBound) / granual);
		String series = Integer.toBinaryString(m);

		BitSet bitset = new BitSet(capacity);
		int idx = capacity - 1;
		for (int i = series.length() - 1; i >= 0; i--) {
			if ('1' == series.charAt(i))
				bitset.set(idx);
			idx--;
		}
		return bitset;
	}

	public Individual getMember(int idx){
		return members[idx];
	}

	public Individual getElitMember(int idx){
		return elitMembers[idx];
	}

	public boolean inherit(Individual child, Individual parent){
		if (parent == null || child == null)
			return false;

		child.setChromosome(parent.getChromosome());
		return true;
	}

	public int getAge(){
		return age;
	}

	public float getCrossRate(){
		return cr;
	}

	public int getSize(){
		return size;
	}
}