package com.horsehour.ml.data;

import java.io.Serializable;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150320
 */
public class RateRecord implements Serializable {
	private static final long serialVersionUID = 1L;
	public float rate;
	public String time;

	public RateRecord(float rate, String tm) {
		this.rate = rate;
		this.time = tm;
	}

	public String toString() {
		return rate + "\t" + time;
	}
}
