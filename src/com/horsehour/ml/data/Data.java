package com.horsehour.ml.data;

import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.math.function.TransferFunction;
import com.horsehour.ml.data.sieve.ARFFSieve;
import com.horsehour.ml.data.sieve.CSVSieve;
import com.horsehour.ml.data.sieve.ColumnSieve;
import com.horsehour.ml.data.sieve.L2RSieve;
import com.horsehour.ml.data.sieve.Sieve;
import com.horsehour.ml.data.sieve.SparseSieve;
import com.horsehour.util.Ace;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * Data Utilities
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Mar. 27, 2013
 */
public class Data {
	/**
	 * @param dataset
	 */
	public static void preprocess(DataSet dataset) {
		int sz = dataset.size();
		SampleSet ss = null;
		for (int i = 0; i < sz; i++) {
			ss = dataset.getSampleSet(i);
			float average = 0;
			List<Integer> labelList = ss.getLabelList();
			for (float label : labelList)
				average += label;

			average /= ss.size();

			boolean tie = true;
			for (int label : labelList) {
				if (label != average) {
					tie = false;
					break;
				}
			}

			if (tie) {
				dataset.removeSampleSet(i);
				i--;
			}
		}
	}

	/**
	 * @param dataset
	 * @param newFeature
	 */
	public static void addFeature(DataSet dataset, float[][] newFeature) {
		SampleSet ss;
		for (int qid = 0; qid < dataset.size(); qid++) {
			ss = dataset.getSampleSet(qid);
			for (int i = 0; i < ss.size(); i++)
				ss.getSample(i).addFeature(newFeature[qid][i]);
		}
	}

	/**
	 * sampling features
	 * 
	 * @param dataset
	 * @param fids
	 * @return new data set based on feature ids
	 */
	public static DataSet sampleFeature(DataSet dataset, int[] fids) {
		SampleSet sampleset;
		for (int i = 0; i < dataset.size(); i++) {
			sampleset = dataset.getSampleSet(i);
			sampleset = sampleFeature(sampleset, fids);
		}
		return dataset;
	}

	/**
	 * 特征抽样
	 * 
	 * @param sampleset
	 * @param fids
	 * @return new sample set based on fids
	 */
	public static SampleSet sampleFeature(SampleSet sampleset, int[] fids) {
		Sample sample;
		for (int i = 0; i < sampleset.size(); i++) {
			sample = sampleset.getSample(i);
			for (int j = 0; j < fids.length; j++)
				sample = new Sample(sample, fids);
		}
		return sampleset;
	}

	/**
	 * Stochastic Disturbance Term
	 * 
	 * @param dataset
	 */
	public static void disturbLabel(DataSet dataset) {
		SampleSet sampleset;
		for (int qid = 0; qid < dataset.size(); qid++) {
			sampleset = dataset.getSampleSet(qid);
			for (int sid = 0; sid < sampleset.size(); sid++) {
				int label = sampleset.getLabel(sid);
				label += MathLib.Rand.uniform(-0.01F, 0.01F);
				sampleset.getSample(sid).setLabel(label);
			}
		}
	}

	/**
	 * @param dataset
	 * @param func
	 * @param newLabels
	 */
	public static Double[][] transLabel(DataSet dataset, TransferFunction func) {
		Double[][] newLabels = new Double[dataset.size()][];
		for (int qid = 0; qid < dataset.size(); qid++) {
			SampleSet sampleset = dataset.getSampleSet(qid);

			int sz = sampleset.size();
			newLabels[qid] = new Double[sz];

			for (int sid = 0; sid < sz; sid++)
				newLabels[qid][sid] = func.calc(sampleset.getLabel(sid));
			MathLib.Matrix.normalize(newLabels[qid]);
		}
		return newLabels;
	}

	/**
	 * Load data set from local file
	 * 
	 * @param src
	 * @param encode
	 * @param sieve
	 * @return data set
	 */
	public static DataSet loadDataSet(String src, String encode, Sieve sieve) {
		BufferedReader br;
		int dim = 0, numSample = 0;

		List<SampleSet> samplesets = new ArrayList<>();
		SampleSet sampleset;
		Sample sample = null;
		String line = "";
		String qid = "";

		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), encode));
			while ((line = br.readLine()) != null) {
				sample = (Sample) sieve.sift(line.trim());
				if (sample == null)
					continue;

				if (qid.equals(sample.getQid())) {
					sampleset = samplesets.get(samplesets.size() - 1);
					sampleset.addSample(sample);
				} else {
					sampleset = new SampleSet();
					sampleset.addSample(sample);
					samplesets.add(sampleset);
					qid = sample.getQid();
				}
				int size = sample.getDim();
				if (size > dim)
					dim = size;
				numSample++;
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return new DataSet(samplesets, dim, numSample);
	}

	public static DataSet loadDataSet(String src, Sieve sieve) {
		return loadDataSet(src, "UTF8", sieve);
	}

	/**
	 * @param src
	 * @param enc
	 * @param delim
	 * @return datum in src
	 */
	public static List<double[]> loadData(String src, String enc, String delim) {
		BufferedReader br;
		int nCol = 0;

		List<double[]> datum = new ArrayList<>();

		String line = "";
		String[] entries = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			while ((line = br.readLine()) != null) {
				entries = line.trim().split(delim);
				nCol = entries.length;

				double[] data = new double[nCol];
				for (int i = 0; i < nCol; i++)
					data[i] = Double.parseDouble(entries[i]);

				datum.add(data);
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		return datum;
	}

	public static List<double[]> loadData(String src, String delim) {
		return loadData(src, "UTF8", delim);
	}

	public static List<double[]> loadData(String src) {
		return loadData(src, "UTF8", "\t");
	}

	public static List<double[]> loadData(File datFile) {
		return loadData(datFile.getAbsolutePath());
	}

	public static List<double[]> loadData(File datFile, String delim) {
		return loadData(datFile.getAbsolutePath(), delim);
	}

	public static List<double[]> loadData(File datFile, String enc, String delim) {
		return loadData(datFile.getAbsolutePath(), enc, delim);
	}

	/**
	 * Load Column Data from Local File
	 * 
	 * @param src
	 * @param enc
	 * @param sieve
	 * @return data set
	 */
	public static StringBuffer loadColumnData(String src, String enc, ColumnSieve sieve) {
		StringBuffer sb = new StringBuffer();
		String line = "";
		try {
			BufferedReader br = null;
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			while ((line = br.readLine()) != null) {
				line = sieve.sift(line.trim());
				if (line == null)
					continue;
				sb.append(line + "\r\n");
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return sb;
	}

	public static StringBuffer loadColumnData(String src, ColumnSieve sieve) {
		return loadColumnData(src, "UTF8", sieve);
	}

	public static SampleSet loadSampleSET(String src, String enc, Sieve sieve) {
		BufferedReader br;

		SampleSet sampleset = new SampleSet();
		Sample sample;
		String line = "";
		int dim = 0;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			while ((line = br.readLine()) != null) {
				sample = (Sample) sieve.sift(line.trim());

				if (sample == null)
					continue;
				int size = sample.getDim();
				if (dim < size)
					dim = size;
				sampleset.addSample(sample);
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		/**
		 * set dimension for each sparse sample
		 */
		for (Sample smp : sampleset.getSamples())
			smp.setDim(dim);

		return sampleset;
	}

	/**
	 * @param src
	 * @param enc
	 * @param lineParser
	 * @return SampleSet
	 */
	public static SampleSet loadSampleSet(String src, String enc, Sieve sieve) {
		BufferedReader br;

		SampleSet sampleset = new SampleSet();

		int dim = 0;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			Sample sample;
			String line = "";

			while ((line = br.readLine()) != null) {
				sample = (Sample) sieve.sift(line.trim());

				if (sample == null)
					continue;

				int size = sample.getDim();

				if (dim < size)
					dim = size;

				sampleset.addSample(sample);
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		/**
		 * set dimension for each sparse sample
		 */
		for (Sample smp : sampleset.getSamples())
			smp.setDim(dim);

		return sampleset;
	}

	public static SampleSet loadSampleSet(String src, Sieve sieve) {
		return loadSampleSet(src, "UTF8", sieve);
	}

	public static SampleSet loadSampleSet(String src) {
		return loadSampleSet(src, "UTF8");
	}

	public static SampleSet loadSampleSet(String src, String enc) {
		Sieve sieve = null;
		BufferedReader br;
		String line = "";
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			while ((line = br.readLine()) != null) {
				if (line.contains("qid:"))
					sieve = new L2RSieve();
				else if (line.startsWith("@") || line.startsWith("%"))
					sieve = new ARFFSieve();
				else if ((line.contains(" ") || line.contains("\t")) && line.contains(":"))
					sieve = new SparseSieve();
				else if (line.contains(","))
					sieve = new CSVSieve();
				break;
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		if (sieve == null) {
			System.err.println("Unknown Format...");
			return null;
		}
		return loadSampleSet(src, enc, sieve);
	}

	/**
	 * Load Rate Set (Recommendation)
	 * 
	 * @param src
	 * @param enc
	 * @return RateSet
	 */
	public static RateSet loadRateSet(String src, String enc) {
		RateSet rs = new RateSet();
		BufferedReader br = null;
		String line;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			while ((line = br.readLine()) != null) {
				String[] subs = line.trim().split("\t");

				int len = subs.length;
				if (len < 2) {
					System.err.println("ERROR: less than two columns.");
					break;
				}

				RateRecord rr = null;// len == 2
				if (len == 3)
					rr = new RateRecord(Float.parseFloat(subs[2]), null);
				else if (len == 4)
					rr = new RateRecord(Float.parseFloat(subs[2]), subs[3]);

				int userId = Integer.parseInt(subs[0]);
				int userIdx = rs.userList.indexOf(userId);
				if (userIdx == -1) {
					rs.userList.add(userId);
					userIdx = rs.userList.size() - 1;// new index

					rs.rateList.add(new ArrayList<Integer>());
					rs.rateHistory.add(new ArrayList<RateRecord>());
					rs.nUser++;
				}

				int itemId = Integer.parseInt(subs[1]);
				int itemIdx = rs.itemList.indexOf(itemId);
				if (itemIdx == -1) {
					rs.itemList.add(itemId);
					itemIdx = rs.itemList.size() - 1;
					rs.nItem++;
				}

				// userIdx > 0, itemIdx > 0
				rs.rateList.get(userIdx).add(itemIdx);
				rs.rateHistory.get(userIdx).add(rr);
				rs.nRate++;
			}
			br.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			return null;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return rs;
	}

	public static RateSet loadRateSet(String src) {
		return loadRateSet(src, "utf-8");
	}

	/**
	 * Split data into m pieces
	 * 
	 * @param src
	 * @param enc
	 * @param m
	 */
	public static void splitData(String src, String enc, int m) {
		BufferedReader br = null;
		List<String> lineList = null;
		int key = -1, preKey = -1;
		String line;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src), enc));
			while ((line = br.readLine()) != null) {
				String[] subs = line.trim().split("\t");
				if (subs.length < 2)
					continue;

				key = Integer.parseInt(subs[0]);
				if (preKey != key) {
					if (preKey != -1)
						randomSplit(lineList, m, src);

					lineList = new ArrayList<String>();
					preKey = key;
				}
				lineList.add(line.trim());
			}
			br.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			return;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	public static void splitData(String src, int m) {
		splitData(src, "UTF8", m);
	}

	public static List<Double> readDataList(String src) {
		List<Double> lines = new ArrayList<>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(src)));
			String line = "";
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty())
					continue;
				else
					lines.add(Double.parseDouble(line));
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return lines;
	}

	/**
	 * Randomly split lines to m files
	 * 
	 * @param lineList
	 * @param m
	 * @param src
	 */
	private static void randomSplit(List<String> lineList, int m, String src) {
		int num = Math.round(lineList.size() / m);
		String dest = new File(src).getParent();

		StringBuffer sb;
		String line;
		for (int i = 0; i < m - 1; i++) {
			Random rand = new Random();
			sb = new StringBuffer();
			int idx;
			for (int j = 0; j < num; j++) {
				idx = rand.nextInt(lineList.size());
				line = lineList.remove(idx);
				sb.append(line + "\r\n");
			}
			try {
				FileUtils.write(new File(dest + "/S" + (i + 1) + ".txt"), "", sb.toString());
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}

		sb = new StringBuffer();
		for (int k = 0; k < lineList.size(); k++) {
			line = lineList.get(k);
			sb.append(line + "\r\n");
		}

		try {
			FileUtils.write(new File(dest + "/S" + m + ".txt"), "", sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * Transform binary digit matrix to row-vector
	 * 
	 * @param src
	 * @return 0-1 string deliminated with Tab
	 */
	public static String getDigitString(String src, String label) {
		List<String> lineList = null;
		try {
			lineList = Files.lines(Paths.get(src)).collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		if (lineList == null || lineList.size() == 0)
			return null;

		StringBuffer sb = new StringBuffer();
		sb.append(label + "\t");
		int len = lineList.get(0).length();
		int count = 0;
		for (String line : lineList) {
			for (int i = 0; i < len; i++) {
				count += i;
				if (line.charAt(i) == '1')
					sb.append((count + 1) + ":1\t");
			}
		}
		int idx = src.indexOf("_");
		src = src.substring(idx + 1).replace(".txt", "").trim();
		sb.append("#" + Integer.parseInt(src) + "\r\n");
		return sb.toString();
	}

	/**
	 * Read Content on Clipboard
	 * 
	 * @return Content on Clipboard
	 * @throws IOException
	 * @throws UnsupportedFlavorException
	 */
	public static String readClipboard() throws UnsupportedFlavorException, IOException {
		Clipboard clip = Toolkit.getDefaultToolkit().getSystemClipboard();
		String content = (String) clip.getData(DataFlavor.stringFlavor);
		return content;
	}

	/**
	 * Read content in terms of list from clipboard
	 * 
	 * @param delim
	 *            used to seperate fields/columns
	 * @return Content in Terms of String List
	 */
	public static List<String> readClipboard(String delim) throws UnsupportedFlavorException, IOException {
		Clipboard clip = Toolkit.getDefaultToolkit().getSystemClipboard();
		String content = (String) clip.getData(DataFlavor.stringFlavor);

		String[] fields = content.split(delim);
		if (fields == null)
			return null;

		List<String> list = new ArrayList<>();
		for (int i = 0; i < fields.length; i++)
			list.add(fields[i]);
		return list;
	}

	public static List<String[]> readClipboard(String delimRow, String delimCol)
			throws UnsupportedFlavorException, IOException {
		Clipboard clip = Toolkit.getDefaultToolkit().getSystemClipboard();
		String content = (String) clip.getData(DataFlavor.stringFlavor);
		String[] fields = content.split(delimRow);
		if (fields == null)
			return null;

		List<String[]> list = new ArrayList<>();
		for (String field : fields)
			list.add(field.split(delimCol));
		return list;
	}

	/**
	 * Related information on the double semi-circle data set is descripted in
	 * the book "Learning from data: a short course" (pp. 109), by Yaser S.
	 * Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin.
	 */
	public static class DoubleSemiCircle {
		/**
		 * Inner Radius and Outer Radius
		 */
		public static double iRad = 10, oRad;
		public static double width = 5, sep = 5;

		static boolean valid(double x1, double x2) {
			double s = x1 * x1 + x2 * x2;
			if (iRad * iRad > s || s > oRad * oRad)
				return false;
			return true;
		}

		public static Pair<double[][], int[]> generate(int nSample) {
			oRad = iRad + width;
			double dX = iRad + .5 * width;
			double dY = -sep;

			double[][] x = new double[2 * nSample][2];
			int[] y = new int[2 * nSample];

			int m = 0;
			while (m < 2 * nSample) {
				double x1 = MathLib.Rand.uniform(-oRad, oRad);
				double x2 = MathLib.Rand.uniform(-oRad, oRad);

				if (!valid(x1, x2))
					continue;

				if (x2 >= 0) {
					y[m] = -1;
				} else {
					x1 += dX;
					x2 += dY;
					y[m] = 1;
				}

				x[m][0] = x1;
				x[m][1] = x2;
				m++;
			}
			return Pair.of(x, y);
		}

		public static SampleSet getSampleSet(int nSample) {
			Pair<double[][], int[]> data = generate(nSample);
			double[][] x = data.getKey();
			int[] y = data.getValue();

			SampleSet sampleset = new SampleSet();
			for (int i = 0; i < 2 * nSample; i++)
				sampleset.addSample(new Sample(x[i], y[i]));
			return sampleset;
		}
	}

	public static double[][] getUnitSquare(int nSample) {
		double[][] x = new double[nSample][2];
		for (int i = 0; i < nSample; i++) {
			x[i][0] = MathLib.Rand.uniform(0, 1);
			x[i][1] = MathLib.Rand.uniform(0, 1);
		}
		return x;
	}

	/**
	 * @param nSample
	 *            number of sample
	 * @param k
	 *            number of center
	 * @param sigma
	 *            bump height
	 * @return
	 */
	public static Map<double[], double[][]> getGaussianMixture(int nSample, int k, double sigma) {
		double[][] muV = new double[k][2];
		for (int i = 0; i < k; i++) {
			muV[i][0] = MathLib.Rand.uniform(0, 1);
			muV[i][1] = MathLib.Rand.uniform(0, 1);
		}

		Map<double[], double[][]> gm = new HashMap<>();
		for (double[] mu : muV) {
			double[][] x = new double[nSample / k][2];
			for (int i = 0; i < nSample / k; i++) {
				x[i][0] += MathLib.Rand.gaussian(mu[0], sigma);
				x[i][1] += MathLib.Rand.gaussian(mu[1], sigma);
			}
			gm.put(mu, x);
		}
		return gm;
	}

	public static void getGaussianMixture(int nSample, int k, double sigma, double[][] x) {
		double[][] muV = new double[k][2];
		for (int i = 0; i < k; i++) {
			muV[i][0] = MathLib.Rand.uniform(0, 1);
			muV[i][1] = MathLib.Rand.uniform(0, 1);
		}

		for (double[] mu : muV) {
			int c = 0;
			for (int i = 0; i < nSample / k; i++) {
				x[c][0] += MathLib.Rand.gaussian(mu[0], sigma);
				x[c][1] += MathLib.Rand.gaussian(mu[1], sigma);
				c++;
			}
		}
	}

	/**
	 * A reader for the MNIST dataset of handwritten digits. It is found at
	 * http://yann.lecun.com/exdb/mnist/.
	 */
	public static StringBuffer getMNIST(String labelFile, String pixelFile) throws IOException {
		DataInputStream labels = null, images = null;
		try {
			labels = new DataInputStream(new FileInputStream(labelFile));
			images = new DataInputStream(new FileInputStream(pixelFile));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		int numMagic = labels.readInt();
		if (numMagic != 2049) {
			System.err.printf("ERROR: Label file should have 2049 (> %d) magic numbers.\n", numMagic);
			System.exit(0);
		}
		numMagic = images.readInt();
		if (numMagic != 2051) {
			System.err.printf("ERROR: Image file should have 2051 (> %d) magic numbers\n", numMagic);
			System.exit(0);
		}

		int numLabels = labels.readInt();
		int numImages = images.readInt();

		int numRows = images.readInt();
		int numCols = images.readInt();

		if (numLabels != numImages) {
			System.err.println("ERROR: Image file and label file do not contain the same number of entries.");
			System.err.println("ERROR: Label file contains: " + numLabels);
			System.err.println("ERROR: Image file contains: " + numImages);
			System.exit(0);
		}

		StringBuffer sb = new StringBuffer();
		int numLabelsRead = 0;
		while (labels.available() > 0 && numLabelsRead < numLabels) {
			for (int colIdx = 0; colIdx < numCols; colIdx++)
				for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
					sb.append(images.readUnsignedByte() + ",");
			sb.append(labels.readUnsignedByte() + "\r\n");
			numLabelsRead++;
		}
		return sb;
	}

	/**
	 * Extract Intensity Features: Average Intensity and Vertical Intensity
	 * Variance
	 * 
	 * @param input
	 *            dim * dim length
	 * @param dim
	 * @return intensity features
	 */
	public static double[] getDigitIntensity(double[] input, int dim) {
		double[] intensity = new double[2];

		int m = 0;
		double[] verticalIntensity = new double[dim];
		for (int i = 0; i < input.length; i++) {
			intensity[0] += input[i];

			verticalIntensity[m] += input[i];
			if ((i + 1) % dim == 0) {
				verticalIntensity[m] /= dim;
				m++;
			}
		}
		intensity[0] /= (dim * dim);

		for (int i = 0; i < dim; i++)
			intensity[1] += Math.pow(verticalIntensity[i] - intensity[0], 2);
		intensity[1] /= (dim - 1);

		return intensity;
	}

	public static class Bridge {
		public static SampleSet getSampleSet(double[][] x, int[] y) {
			SampleSet sampleset = new SampleSet();
			for (int i = 0; i < y.length; i++)
				sampleset.addSample(new Sample(x[i], y[i]));
			return sampleset;
		}

		public static Pair<double[][], int[]> getSamples(SampleSet sampleset) {
			int n = sampleset.size();
			int m = sampleset.dim();
			double[][] x = new double[n][m];
			int[] y = new int[n];

			Sample sample;
			for (int i = 0; i < n; i++) {
				sample = sampleset.getSample(i);
				x[i] = sample.getFeatures();
				y[i] = sample.getLabel();
			}
			return Pair.of(x, y);
		}

		public static void getSamples(SampleSet sampleset, double[][] x, int[] y) {
			Sample sample;
			for (int i = 0; i < sampleset.size(); i++) {
				sample = sampleset.getSample(i);
				x[i] = sample.getFeatures();
				y[i] = sample.getLabel();
			}
		}
	}

	public static class Reshape {
		/**
		 * Relabel samples from index. For example index = 0, samples will be
		 * assigned 0, 1, 2, ... based on their original labels
		 * 
		 * @param sampleset
		 * @param index
		 */
		public static Map<Integer, Integer> reLabel(SampleSet sampleset, int index) {
			List<Integer> uniqLabel = sampleset.getUniqueLabels();
			uniqLabel.sort((a, b) -> Integer.compare(a, b));

			Map<Integer, Integer> map = new HashMap<>();
			for (int i = 0; i < uniqLabel.size(); i++)
				map.put(uniqLabel.get(i), index + i);

			Sample sample;
			for (int i = 0; i < sampleset.size(); i++) {
				sample = sampleset.getSample(i);
				sample.setLabel(map.get(sample.getLabel()));
			}
			return map;
		}

		/**
		 * Relabel samples based on specific map-table
		 * 
		 * @param sampleset
		 * @param map
		 */
		public static void reLabel(SampleSet sampleset, Map<Integer, Integer> map) {
			Sample sample;
			for (int i = 0; i < sampleset.size(); i++) {
				sample = sampleset.getSample(i);
				int label = sample.getLabel();
				if (map.containsKey(label))
					sample.setLabel(map.get(label));
			}
		}

		/**
		 * ReLabel samples using a function
		 * 
		 * @param sampleset
		 * @param f
		 */
		public static void reLabel(SampleSet sampleset, Function<Integer, Integer> f) {
			Sample sample;
			for (int i = 0; i < sampleset.size(); i++) {
				sample = sampleset.getSample(i);
				int label = sample.getLabel();
				sample.setLabel(f.apply(label));
			}
		}

		public static Double[][] reLabel(DataSet dataset, Function<Integer, Double> f) {
			Double[][] labels = new Double[dataset.size()][];
			for (int qid = 0; qid < dataset.size(); qid++) {
				SampleSet sampleset = dataset.getSampleSet(qid);

				int sz = sampleset.size();
				labels[qid] = new Double[sz];

				for (int i = 0; i < sz; i++)
					labels[qid][i] = f.apply(sampleset.getLabel(i));
			}
			return labels;
		}

		public static void reLabel(int[] y, Function<Integer, Integer> f) {
			for (int i = 0; i < y.length; i++)
				y[i] = f.apply(y[i]);
		}

		public static void reLabel(int[] y, int[] values) {
			List<Integer> distinct = MathLib.Data.distinct(y);
			if (distinct.size() != values.length) {
				throw new IllegalArgumentException("Dimensional dismatch for given values and distinct labels.");
			}
			Collections.sort(distinct);
			for (int i = 0; i < y.length; i++) {
				int index = distinct.indexOf(y[i]);
				y[i] = values[index];
			}
		}

		public static void reLabel(int[] y, int index) {
			List<Integer> distinct = MathLib.Data.distinct(y);
			Collections.sort(distinct);
			for (int i = 0; i < y.length; i++) {
				int d = distinct.indexOf(y[i]);
				y[i] = index + d;
			}
		}

		public static void reLabel(int[] y, Map<Integer, Integer> map) {
			for (int i = 0; i < y.length; i++)
				y[i] = map.get(y[i]);
		}

		public static SampleSet shuffle(SampleSet sampleset) {
			List<Sample> samples = sampleset.getSamples();
			Collections.shuffle(samples);
			return new SampleSet(samples);
		}

		/**
		 * @param sampleset
		 * @param fid
		 * @param ascend
		 * @return sorted sample set in order
		 */
		public static void sort(SampleSet sampleset, int fid, boolean ascend) {
			if (ascend)
				sampleset.getSamples().sort((s1, s2) -> Double.compare(s1.getFeature(fid), s2.getFeature(fid)));
			else
				sampleset.getSamples().sort((s1, s2) -> Double.compare(s2.getFeature(fid), s1.getFeature(fid)));
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		Ace ace = new Ace("");

		Map<double[], double[][]> gm = Data.getGaussianMixture(1000, 10, 0.1);
		List<String> labelList = new ArrayList<>();
		double[][] data = new double[1000][2];

		int c = 0;
		for (Entry<double[], double[][]> entry : gm.entrySet()) {
			double[] mu = entry.getKey();
			for (double[] d : entry.getValue()) {
				labelList.add(Arrays.toString(mu));
				data[c++] = d;
			}
		}
		ace.scatter("x1", "x2", labelList, data);

		TickClock.stopTick();
	}
}
