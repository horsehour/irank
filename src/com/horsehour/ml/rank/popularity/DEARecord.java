package com.horsehour.ml.rank.popularity;

/**
 * DEA method separates the features of the observations in input and output; 
 * The DeaRecord is a wrapper Object which stores the input and output parts of our
 * data points.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class DEARecord {
	/**
	 * The values of features that are considered as input. It can be empty if 
	 * if no input is assumed. 
	 */
	private final double[] input;

	/**
	 * The values of features that are considered as output.
	 */
	private final double[] output;

	/**
	 * Constructor of DEA Record with only output
	 * 
	 * @param output    The output part of our data record
	 */
	public DEARecord(double[] output) {
		this.output = output;
		this.input = new double[0];
	}

	/**
	 * Constructor of DEA Record with both input and output
	 * 
	 * @param output    The output part of our data record
	 * @param input     The input part of our data record
	 */
	public DEARecord(double[] output, double[] input) {
		this.output = output;
		this.input = input;
	}

	/**
	 * Getter for input
	 * 
	 * @return  Returns the input part of our data.
	 */
	public double[] getInput(){
		return input;
	}

	/**
	 * Getter for output
	 * 
	 * @return  Returns the output part of our data.
	 */
	public double[] getOutput(){
		return output;
	}
}
