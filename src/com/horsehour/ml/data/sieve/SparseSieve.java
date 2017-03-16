package com.horsehour.ml.data.sieve;

import java.util.HashMap;
import java.util.Map;

import com.horsehour.ml.data.Sample;

/**
 * Parse samples in the format as follows:
 * <p>
 * +1 1:0.7 2:0.2 5:0.1 13:-1</br>
 * -1 2:0.9 5:0.3 9:0.7 15:8</br>
 * ... ... </br></p>
 * where the first column is the class label, and the following
 * columns are indexed features.</p>
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130509
 */
public class SparseSieve extends Sieve {
	public SparseSieve() {}

	@Override
	public Sample sift(String line) {
		int idx = line.indexOf('#');
		String meta = "";
		if(idx == 0)
			return null;
		
		if(idx == -1)
			idx = line.length();
		else
			meta = line.substring(idx).trim();

		line = line.substring(0, idx).trim();
		line = line.replace("  ", "");
		String[] segments = line.split(" |\t");

		Map<Integer, Double> featureList = new HashMap<>();

		String[] pair = null;
		for (int i = 1; i < segments.length; i++) {
			pair = segments[i].split(":");
			featureList.put(Integer.parseInt(pair[0]) - 1, Double.parseDouble(pair[1]));
		}

		int label = (int) Double.parseDouble(segments[0].replace("+", ""));
		Sample sample = new Sample(featureList, label);
		sample.setDim(Integer.parseInt(pair[0]));//default, last index
		sample.setMeta(meta);
		return sample;
	}
}
