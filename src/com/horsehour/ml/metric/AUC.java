package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

/**
 * Area Under Curve of ROC
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131125
 */
public class AUC extends Metric {

	public AUC() {}

	@Override
	public double measure(List<? extends Number> desireList, List<? extends Number> predictList){
		int nPositive = 0;
		int nNegative = 0;

		int len = predictList.size();
		List<Pair<Double, Integer>> list = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			int label = desireList.get(i).intValue();
			if (label == 0)
				nNegative++;

			list.add(Pair.of(predictList.get(i).doubleValue(), label));
		}

		nPositive = len - nNegative;

		Collections.sort(list);

		float fp = 0;
		float tp = 0;
		float fpPrev = 0;
		float tpPrev = 0;
		double area = 0;
		double fPrev = Double.MIN_VALUE;

		Pair<Double, Integer> pair;
		for (int i = 0; i < len; i++) {
			pair = list.get(i);

			double curF = pair.getKey();

			if (curF != fPrev) {
				area += Math.abs(fp - fpPrev) * ((tp + tpPrev) / 2.0);
				fPrev = curF;
				fpPrev = fp;
				tpPrev = tp;
			}

			int label = pair.getValue();
			if (label == 1)
				tp++;
			else
				fp++;
		}

		area += Math.abs(nNegative - fpPrev) * ((nPositive + tpPrev) / 2.0);
		area /= (1.0d * nPositive * nNegative);
		return area;
	}

	public static void main(String[] args){
		List<Double> predict = new ArrayList<Double>();
		List<Integer> label = new ArrayList<Integer>();

		predict.add(1.6);
		predict.add(2.5);
		predict.add(3.3);
		predict.add(3.3);
		label.add(0);
		label.add(1);
		label.add(0);
		label.add(0);

		System.out.println(new AUC().measure(label, predict));
	}
}
