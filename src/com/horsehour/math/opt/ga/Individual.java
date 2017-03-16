package com.horsehour.math.opt.ga;

import java.util.BitSet;

/**
 * 遗传算法基本单位
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Dec. 18 2012
 */
public class Individual {
	private BitSet chromosome;// consist of genes
	private int capacity;

	public Individual() {}

	public Individual(int capacity) {
		this.capacity = capacity;
		chromosome = new BitSet(capacity);
	}

	public Individual(int capacity, BitSet chr) {
		this(capacity);
		this.chromosome = chr;
	}

	/**
	 * one-point cross over
	 * 
	 * @param idv
	 *            another individual that will be crossed over
	 * @param fromIdx
	 *            start point for cross over
	 * @param toIdx
	 *            end point for cross over
	 */
	public void onepoint(Individual idv, int fromIdx, int toIdx){
		if (toIdx <= fromIdx)
			return;

		for (int i = fromIdx; i <= toIdx; i++) {
			boolean gene1 = chromosome.get(i);
			boolean gene2 = idv.chromosome.get(i);

			if (gene1 ^ gene2)
				if (gene1) {
					chromosome.clear(i);
					idv.chromosome.set(i);
				} else {
					chromosome.set(i);
					idv.chromosome.clear(i);
				}
		}
	}

	public BitSet getChromosome(){
		return chromosome;
	}

	public void setChromosome(BitSet bitset){
		chromosome.clear();
		for (int i = 0; i < capacity; i++)
			if (bitset.get(i))
				chromosome.set(i);
	}

	public int getCapacity(){
		return capacity;
	}
}
