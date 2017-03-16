package com.horsehour.ml.classifier.svm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.math.function.KernelFunction;
import com.horsehour.math.function.LinearKernel;
import com.horsehour.math.function.RBFKernel;
import com.horsehour.ml.classifier.OvOMultiClassifier;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.scale.DataScale;
import com.horsehour.ml.data.scale.IntervalScale;
import com.horsehour.ml.model.Model;
import com.horsehour.ml.model.SVMModel;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * SMO(Sequential Minimal Optimization)
 * 
 * <li><cite>Platt J C. Sequential Minimal Optimization: A Fast Algorithm for
 * Training Support Vector Machines. 1998.</cite></li>
 * 
 * <li><cite>Keerthi S S, Shevade S K, Bhattacharyya C, et al. Improvements to
 * Platt's SMO Algorithm for SVM Classifier Design[J]. Neural Computation, 1999,
 * 13(3):637 - 649.</cite></li>
 * 
 * <li><cite>Fan R E, Chen P H, Lin C J. Working set selection using second
 * order information for training support vector machines[J]. The Journal of
 * Machine Learning Research, 2005, 6: 1889-1918.</cite></li>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Apr 17 2014
 */
public class SMO extends OvOMultiClassifier {
	public KernelFunction kernel = new LinearKernel();

	public SampleSet biTrainset;
	public double[][] gram;

	public float c = 5;
	public float epsilon = 1.0E-5F;
	public float tolerance = 1.0E-5F;
	public int nTrain;

	public double[] alpha;
	public double bias = 0;

	public List<Integer> nonbound;// i - 0 < alpha[i] < C

	public Model model;

	private final Random random = new Random();

	public SMO() {
	}

	/**
	 * learn the optimal model in analytic way
	 * <p>
	 * kkt conditions
	 * <li>ai = 0 => yi*f(xi) >= 1 (correctly classified: ri = yi*ei = yi*f(xi)
	 * - yi*yi >=0)</li>
	 * <li>0 < ai < C => yi*f(xi) = 1 (svs on the margin: ri = 0)</li>
	 * <li>ai = C => yi*f(xi) <= 1 (misclassified: ri <= 0)</li>
	 * </p>
	 */
	@Override
	public Model learn(Pair<Integer, Integer> pair) {
		biTrainset = new SampleSet();
		int c1 = pair.getKey(), c2 = pair.getValue();
		for (int id : cluster.get(c1))
			biTrainset.addSample(trainset.getSample(id));
		for (int id : cluster.get(c2))
			biTrainset.addSample(trainset.getSample(id));

		nTrain = biTrainset.size();
		alpha = new double[nTrain];
		nonbound = new ArrayList<Integer>();
		nIter = 10;// 拉格朗日乘子保持不变的迭代次数上限
		initKernelMatrix();

		int iter = 0;
		while (iter < nIter) {
			int count = 0;// 乘子改变的次数(成对改变)

			for (int i = 0; i < nTrain; i++) {
				double e1 = trainError(i);

				// 1st sample: violate kkt conditions
				if (violateKKT(i, e1)) {

					// 2ed sample
					int j = -1;
					double eta = 0;
					while (eta <= 0) {// i,j两个样本相同,核函数不满足Mercer条件
						j = searchNext(i, e1);
						eta = gram[i][i] + gram[j][j] - 2 * gram[i][j];// non-negative
					}

					double lowBound, highBound;
					if (biTrainset.getLabel(i) == biTrainset.getLabel(j)) {
						lowBound = Math.max(0, alpha[i] + alpha[j] - c);
						highBound = Math.min(0, alpha[i] + alpha[j]);
					} else {
						lowBound = Math.max(0, alpha[j] - alpha[i]);
						highBound = Math.min(c, alpha[j] - alpha[i] + c);
					}

					double e2 = trainError(j);

					double a2 = alpha[j];
					alpha[j] += biTrainset.getLabel(j) * (e1 - e2) / eta;
					extendNonbound(j);

					if (alpha[j] < lowBound)
						alpha[j] = lowBound;
					else if (alpha[j] > highBound)
						alpha[j] = highBound;

					if (Math.abs(alpha[j] - a2) < tolerance)
						continue;

					double a1 = alpha[i];
					alpha[i] += biTrainset.getLabel(i) * biTrainset.getLabel(j) * (a2 - alpha[j]);
					extendNonbound(i);

					updateBias(i, a1, e1, j, a2, e2);

					count++;
				}
			}

			if (count == 0)// 乘子对不变
				iter++;
			else
				iter = 0;
		}
		updateModel();
		return model;
	}

	/**
	 * initialize the kernel matrix
	 */
	private void initKernelMatrix() {
		gram = new double[nTrain][nTrain];

		double[] xi = null;
		for (int i = 0; i < nTrain; i++) {
			xi = biTrainset.getSample(i).getFeatures();
			for (int j = i; j < nTrain; j++) {
				if (j == i)
					gram[i][j] = kernel.calc(xi, xi);
				else
					gram[j][i] = gram[i][j] = kernel.calc(xi, biTrainset.getSample(j).getFeatures());
			}
		}
	}

	/**
	 * 当前训练模型在样本i上的预测误差
	 * 
	 * @param idx
	 * @return prediction error on given training sample
	 */
	private double trainError(int idx) {
		double predict = 0;
		for (int i = 0; i < nTrain; i++)
			predict += alpha[i] * biTrainset.getLabel(i) * gram[i][idx];

		return predict + bias - biTrainset.getLabel(idx);
	}

	/**
	 * search the second alpha
	 * 
	 * @param idx
	 * @param error
	 * @return index of the second alpha
	 */
	private int searchNext(int idx, double error) {
		int next = idx;
		if (nonbound.isEmpty()) {// 支持向量为空
			while (next == idx)
				next = random.nextInt(nTrain);
			return next;
		}

		double max = 0;
		int size = nonbound.size();
		for (int k = 0; k < size; k++) {
			double delta = Math.abs(error - trainError(k));

			if (delta > max) {
				max = delta;
				next = k;
			}
		}
		return next;
	}

	/**
	 * 判定是否违反KKT条件
	 * 
	 * @param idx
	 * @param error
	 * @return {@code true} if violate kkt conditions
	 */
	private boolean violateKKT(int idx, double error) {
		double ri = biTrainset.getLabel(idx) * error;
		boolean ret = (alpha[idx] < c && ri + epsilon < 0);
		ret = ret || (alpha[idx] > 0 && ri > epsilon);
		return ret;
	}

	/**
	 * 扩展nonbound
	 * 
	 * @param idx
	 */
	private void extendNonbound(int idx) {
		if (0 < alpha[idx] && alpha[idx] < c) {
			if (nonbound.contains(idx))
				return;
			nonbound.add(idx);
		}
	}

	/**
	 * update bias of model
	 * 
	 * @param idx1
	 * @param prev1
	 * @param e1
	 * @param idx2
	 * @param prev2
	 * @param e2
	 */
	private void updateBias(int idx1, double prev1, double e1, int idx2, double prev2, double e2) {
		double b1 = bias - e1 - biTrainset.getLabel(idx1) * (alpha[idx1] - prev1) * gram[idx1][idx1]
				- biTrainset.getLabel(idx2) * (alpha[idx2] - prev2) * gram[idx1][idx2];
		double b2 = bias - e2 - biTrainset.getLabel(idx1) * (alpha[idx1] - prev1) * gram[idx1][idx2]
				- biTrainset.getLabel(idx2) * (alpha[idx2] - prev2) * gram[idx2][idx2];
		bias = (b1 + b2) / 2;
	}

	/**
	 * update model
	 */
	private void updateModel() {
		List<Sample> svs = new ArrayList<Sample>();
		List<Double> alphaSV = new ArrayList<Double>();

		for (int i = 0; i < nTrain; i++) {
			if (alpha[i] == 0)
				continue;

			svs.add(biTrainset.getSample(i));
			alphaSV.add(alpha[i]);
		}
		model = new SVMModel(alphaSV, svs, bias);
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		SMO smo = new SMO();
		smo.kernel = new RBFKernel();

		String data = "Data/classification/iris.dat";

		SampleSet sampleset = Data.loadSampleSet(data);
		DataScale dataScale = new IntervalScale(0, 1);
		dataScale.scale(sampleset);

		int maxIter = 1;
		List<SampleSet> splitList;
		List<Double> perfList = new ArrayList<>();
		for (int iter = 0; iter < maxIter; iter++) {
			splitList = sampleset.splitSamples(new float[] { 0.6F, 0.4F });
			smo.trainset = splitList.get(0);
			smo.learn();
			perfList.add(smo.eval(splitList.get(1)));
		}

		System.out.println("Avg Test Perf (" + maxIter + ") = " + MathLib.Data.mean(perfList));

		TickClock.stopTick();
	}
}