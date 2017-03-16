package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;

import com.horsehour.util.MathLib;

/**
 * <p>
 * Various measures of complexity were developed to compare time series and
 * distinguish regular, (e.g. periodic), chaotic and random behavior... The main
 * types of complexity parameters are entropies, fractal dimensions, Lyapunov
 * exponents. They are all defined for typical orbits of presumably ergodic
 * dynamical systems, and there are profound relations between these quantities
 * which allow to compute one from the other.
 * <p>
 * The basic conceptual problem is that these definitions are not made for an
 * arbitrary series of observations {x1,x2…}. As a consequence, there is also a
 * computational problem.
 * <p>
 * Permutation entropy provides a simple and robust method to estimate
 * complexity of time series, taking the temporal order of the values into
 * account. Furthermore, permutation entropy can be used to determine embedding
 * parameters or identify couplings between time series.
 * <p>
 * For further information please consider the reference: Permutation Entropy –
 * a natural complexity measure for time series.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2014年5月16日 下午1:50:35
 **/
public class PermuEntropy {
	public double measure(List<Integer> list, int order){
		if (list == null || list.size() == 0) {
			System.err.println("It is an empty list");
			return -1;
		}

		int size = list.size();
		if (order <= 1) {
			System.err.println("Order should be a natural value larger than 1");
			return -1;
		} else if (order > size) {
			System.err.println("Order should be not be greater than the size of given list");
			return -1;
		}

		List<Double> probList = new ArrayList<Double>();
		Map<String, Integer> patternTable = new HashMap<String, Integer>();
		int nPermu = size - order + 1;
		for (int i = 0; i < nPermu; i++) {
			int[] rank = MathLib.getRank(list.subList(i, i + order), true);
			String pattern = StringUtils.join(rank, "\t");
			if (patternTable.containsKey(pattern))
				patternTable.put(pattern, patternTable.get(pattern) + 1);
			else
				patternTable.put(pattern, 1);
		}
		for (String pattern : patternTable.keySet())
			probList.add(1.0D * patternTable.get(pattern) / nPermu);
		return Entropy.measure(probList);
	}

	public String name(){
		return "PermutationEntropy";
	}
}
