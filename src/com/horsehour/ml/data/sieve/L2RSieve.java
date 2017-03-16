package com.horsehour.ml.data.sieve;

import java.util.HashMap;
import java.util.Map;

import com.horsehour.ml.data.Sample;

/**
 * 解析如下格式的数据:
 * 
 * <p>0 qid:167 1:0.34 2:1.0 ... 45:0.20 ...</p>
 * <p>1 qid:167 1:0.71 5:0.05 ... 45:0.10 ...</p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130327
 */
public class L2RSieve extends Sieve {

	@Override
	public Sample sift(String line) {
		int idx = line.indexOf('#');
		String meta = "";
		if(idx == -1)
			idx = line.length();
		else
			meta = line.substring(idx).trim();

		line = line.substring(0, idx).trim();
		String[] segments = line.split(" ");

		int label = Integer.parseInt(segments[0]);
		String qid = segments[1].split(":")[1];

		Map<Integer, Double> featureList = new HashMap<>();

		String key = "", val = "";
		for (int i = 2; i < segments.length; i++) {
			key = segments[i].split(":")[0];
			val = segments[i].split(":")[1];
			featureList.put(Integer.parseInt(key) - 1, Double.parseDouble(val));
		}
		Sample sample = new Sample(featureList, label, qid);
		sample.setDim(Integer.parseInt(key));//default, last index
		sample.setMeta(meta);
		return sample;
	}
}
