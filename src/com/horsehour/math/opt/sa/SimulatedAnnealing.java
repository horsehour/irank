package com.horsehour.math.opt.sa;

import java.util.Random;

/**
 * Simulated annealing algorithm
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130714
 */
public class SimulatedAnnealing {
	public int iter = 100;
	public double temperature = 10000.0;
	public double minTemperature = 0.00001;
	public double boltzman = 0.9999;

	protected ParticleState previousState;
	protected ParticleState disturbedState;

	public SimulatedAnnealing() {
	}

	public void init(ParticleState state) {
		previousState = state;
	}

	/**
	 * @param state
	 * @return 给定粒子状态下的能量
	 */
	public double calcEnergy(ParticleState state) {
		return 0;
	}

	/**
	 * 粒子状态发生扰动
	 * 
	 * @param initState
	 * @param initEnergy
	 *            初始状态的能量
	 * @return 扰动引发的状态能量改变
	 */
	public double disturbState(ParticleState initState, double initEnergy) {
		// 最佳选择-根据目标函数，确定增量方式改变
		disturbedState = null;
		return calcEnergy(disturbedState) - initEnergy;
	}

	/**
	 * Accept and update to new state
	 * 
	 * @param state
	 */
	public void acceptState(ParticleState state) {
		previousState = state;
	}

	public void simulate() {
		double energy = calcEnergy(previousState);
		double energyDiff = 0;
		double prob = 0;

		Random rand = new Random();
		boolean accept = false;

		while ((temperature > minTemperature) || (iter > 0)) {
			energyDiff = disturbState(previousState, energy);
			// TODO: does it need to use the Boltzman constant
			prob = Math.exp(-energyDiff / (boltzman * temperature));
			accept = ((energyDiff < 0) || (energyDiff * (prob - rand.nextDouble()) >= 0));
			if (accept) {
				acceptState(disturbedState);
				energy += energyDiff;
			}
			// decrease the temperature
			temperature *= boltzman;
			iter--;
		}
	}

	class ParticleState {

	};
}
