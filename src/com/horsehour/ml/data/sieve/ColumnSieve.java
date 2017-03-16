package com.horsehour.ml.data.sieve;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20160408
 */
public class ColumnSieve extends Sieve {

	public String sift(String line) {
		if (line.contains("China,CHN,")) {
			if (line.contains("490,\"Other Asia, nes\""))
				return line.replace("\"Other Asia, nes\"", "Taiwan");
			return line;
		}
		return null;
	}
}
