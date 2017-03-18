package com.horsehour.util;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BiPredicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.axis.AxisLocation;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.SymbolAxis;
import org.jfree.chart.labels.PieSectionLabelGenerator;
import org.jfree.chart.labels.StandardPieSectionLabelGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.CombinedDomainXYPlot;
import org.jfree.chart.plot.CombinedRangeCategoryPlot;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.LookupPaintScale;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.renderer.xy.XYShapeRenderer;
import org.jfree.chart.title.PaintScaleLegend;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DatasetUtilities;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.xy.DefaultXYZDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.ShapeUtilities;

/**
 * Data Analysis CEnter (Ace) based on JFreeChart. Another important extension
 * of JFreeChart is SSJ for statistical simultation. See
 * http://www.iro.umontreal.ca/~simardr/ssj/indexe.html
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Apr. 16, 2013
 */
public class Ace {
	String destFile = "";
	ApplicationFrame frame;

	int width = 1024, height = 640;

	public Ace(String title) {
		frame = new ApplicationFrame(title);
	}

	public Ace(String title, String dest) {
		frame = new ApplicationFrame(title);
		this.destFile = dest;
	}

	/**
	 * @see http://stackoverflow.com/questions/309149/generate-distinctly-
	 *      different-rgb-colors-in-graphs
	 *
	 */
	public static class RainBow {
		public static Color[] getColors(int n) {
			Color[] colors = new Color[n];
			for (int i = 0; i < n; i++)
				colors[i] = getColor(i);
			return colors;
		}

		public static Color getColor(int i) {
			return new Color(getRGB(i));
		}

		public static int getRGB(int index) {
			int[] p = getPattern(index);
			return getElement(p[0]) << 16 | getElement(p[1]) << 8 | getElement(p[2]);
		}

		static int getElement(int index) {
			int value = index - 1;
			int v = 0;
			for (int i = 0; i < 8; i++) {
				v = v | (value & 1);
				v <<= 1;
				value >>= 1;
			}
			v >>= 1;
			return v & 0xFF;
		}

		static int[] getPattern(int index) {
			int n = (int) Math.cbrt(index);
			index -= (n * n * n);
			int[] p = new int[3];
			Arrays.fill(p, n);
			if (index == 0) {
				return p;
			}
			index--;
			int v = index % 3;
			index = index / 3;
			if (index < n) {
				p[v] = index % n;
				return p;
			}
			index -= n;
			p[v] = index / n;
			p[++v % 3] = index % n;
			return p;
		}
	}

	void show(JFreeChart chart) {
		if (destFile.isEmpty()) {
			ChartPanel panel = new ChartPanel(chart);
			panel.setPreferredSize(new Dimension(width, height));
			frame.setContentPane(panel);
			frame.pack();
			RefineryUtilities.centerFrameOnScreen(frame);
			frame.setVisible(true);
		} else {
			if (destFile.toLowerCase().endsWith(".jpg"))
				try {
					ChartUtilities.saveChartAsJPEG(new File(destFile), chart, width, height);
				} catch (IOException e) {
					e.printStackTrace();
				}
			else if (destFile.toLowerCase().endsWith(".png"))
				try {
					ChartUtilities.saveChartAsPNG(new File(destFile), chart, width, height);
				} catch (IOException e) {
					e.printStackTrace();
				}
		}
	}

	public void pie(List<String> labels, List<? extends Number> data) {
		DefaultPieDataset dataset = new DefaultPieDataset();
		for (int i = 0; i < labels.size(); i++)
			dataset.setValue(labels.get(i), data.get(i));

		JFreeChart chart = ChartFactory.createPieChart("", dataset, true, true, false);
		PieSectionLabelGenerator gen = new StandardPieSectionLabelGenerator("{0}: {1} ({2})", new DecimalFormat("0"),
				new DecimalFormat("0%"));

		PiePlot plot = (PiePlot) chart.getPlot();
		plot.setLabelGenerator(gen);

		show(chart);
	}

	public void pie(List<String> labels, double[] data) {
		DefaultPieDataset dataset = new DefaultPieDataset();
		for (int i = 0; i < labels.size(); i++)
			dataset.setValue(labels.get(i), data[i]);

		JFreeChart chart = ChartFactory.createPieChart("", dataset, true, true, false);
		PieSectionLabelGenerator gen = new StandardPieSectionLabelGenerator("{0}: {1} ({2})", new DecimalFormat("0"),
				new DecimalFormat("0%"));

		PiePlot plot = (PiePlot) chart.getPlot();
		plot.setLabelGenerator(gen);

		show(chart);
	}

	public void histogram(double[][] data, int bins) {
		HistogramDataset dataset = new HistogramDataset();
		int len = data.length, dim = data[0].length;
		for (int k = 0; k < dim; k++) {
			double[] dat = new double[len];
			for (int i = 0; i < len; i++)
				dat[i] = data[i][k];
			dataset.addSeries("dim-" + k, dat, bins);
		}
		JFreeChart chart = ChartFactory.createHistogram("", "", "", dataset, PlotOrientation.VERTICAL, true, true,
				false);

		XYPlot plot = chart.getXYPlot();
		XYBarRenderer renderer = (XYBarRenderer) plot.getRenderer();
		renderer.setBarPainter(new StandardXYBarPainter());

		NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		show(chart);
	}

	public void bar(List<String> rowLabels, List<String> columnLabels, String xLabel, String yLabel, double[][] data) {
		DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int i = 0; i < rowLabels.size(); i++) {
			for (int j = 0; j < columnLabels.size(); j++)
				dataset.addValue(data[i][j], rowLabels.get(i), columnLabels.get(j));
		}

		JFreeChart chart = ChartFactory.createBarChart("", xLabel, yLabel, dataset, PlotOrientation.VERTICAL, true,
				true, false);

		CategoryPlot plot = chart.getCategoryPlot();

		// NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		// axis.setAutoRange(true);
		// axis.setAutoRangeIncludesZero(false);
		double[] bound = getBound(dataset);
		plot.getRangeAxis().setRange(bound[0] * 0.99, bound[1]);

		BarRenderer renderer = new BarRenderer();
		renderer.setBarPainter(new StandardBarPainter());
		renderer.setShadowVisible(false);
		renderer.setItemMargin(0.01);// margin between bars

		plot.setRenderer(renderer);

		show(chart);

	}

	public void bar(List<String> rowLabels, List<String> columnLabels, double[][] data) {
		bar(rowLabels, columnLabels, "", "", data);
	}

	public void bar(List<String> rowLabels, List<String> columnLabels, List<List<Double>> data) {
		DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int i = 0; i < rowLabels.size(); i++) {
			for (int j = 0; j < columnLabels.size(); j++)
				dataset.addValue(data.get(i).get(j), rowLabels.get(i), columnLabels.get(j));
		}
		double[] bound = getBound(dataset);

		JFreeChart chart = ChartFactory.createBarChart("", "", "", dataset, PlotOrientation.VERTICAL, true, true,
				false);

		CategoryPlot plot = chart.getCategoryPlot();

		// NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		// axis.setAutoRange(true);
		// axis.setAutoRangeIncludesZero(false);
		plot.getRangeAxis().setRange(bound[0] * 0.99, bound[1]);

		BarRenderer renderer = new BarRenderer();
		renderer.setBarPainter(new StandardBarPainter());
		renderer.setShadowVisible(false);
		renderer.setItemMargin(0.01);// margin between bars
		plot.setRenderer(renderer);

		show(chart);
	}

	DefaultCategoryDataset createCategoryDataset(List<String> rowLabels, List<String> columnLabels, double[][] data) {
		DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				dataset.addValue(data[i][j], rowLabels.get(i), columnLabels.get(j));
		return dataset;
	}

	double[] getBound(DefaultCategoryDataset dataset) {
		double[] bound = { Double.MAX_VALUE, Double.MIN_VALUE };
		for (int i = 0; i < dataset.getRowCount(); i++) {
			for (int j = 0; j < dataset.getColumnCount(); j++) {
				double val = dataset.getValue(i, j).doubleValue();
				if (val > bound[1])
					bound[1] = val;
				else if (val < bound[0])
					bound[0] = val;
			}
		}
		return bound;
	}

	/**
	 * Combine bars with the same range axis
	 * 
	 * @param rowLabels
	 * @param columnLabels
	 * @param xLabel
	 * @param yLabel
	 * @param data
	 */
	public void combinedRangeBars(List<List<String>> rowLabels, List<List<String>> columnLabels, String xLabel,
			String yLabel, List<double[][]> data) {

		NumberAxis axis = new NumberAxis(xLabel);
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		CombinedRangeCategoryPlot plot = new CombinedRangeCategoryPlot(axis);

		BarRenderer renderer = new BarRenderer();
		renderer.setBarPainter(new StandardBarPainter());
		renderer.setShadowVisible(false);
		renderer.setItemMargin(0.01);// margin between bars

		LegendItemCollection legends = null;

		double[] bound = { Double.MAX_VALUE, Double.MIN_VALUE };
		int nSubplot = data.size();
		for (int i = 0; i < nSubplot; i++) {
			CategoryPlot subplot = new CategoryPlot();
			subplot.setDomainAxis(new CategoryAxis());
			subplot.setRangeAxis(new NumberAxis());
			// margin bettween different categories
			subplot.getDomainAxis().setCategoryMargin(0.1);
			subplot.getDomainAxis()
					.setCategoryLabelPositions(CategoryLabelPositions.createUpRotationLabelPositions(Math.PI / 6.0));
			DefaultCategoryDataset dataset = createCategoryDataset(rowLabels.get(i), columnLabels.get(i), data.get(i));
			double[] bnd = getBound(dataset);
			if (bound[0] > bnd[0])
				bound[0] = bnd[0];
			if (bound[1] < bnd[1])
				bound[1] = bnd[1];

			subplot.setDataset(dataset);
			subplot.setRenderer(renderer);

			if (i == nSubplot - 1)
				legends = subplot.getLegendItems();

			subplot.setFixedLegendItems(null);

			plot.add(subplot);
		}

		plot.getRangeAxis().setLabel(yLabel);
		plot.getRangeAxis().setRange(bound[0] * 0.99, bound[1]);

		plot.setFixedLegendItems(legends);

		JFreeChart chart = new JFreeChart("", plot);
		show(chart);
	}

	public void line(String xLabel, String yLabel, double[][] data) {
		XYSeries series = new XYSeries("");
		int len = data.length;
		for (int i = 0; i < len; i++) {
			series.add(data[i][0], data[i][1]);
		}

		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(series);
		JFreeChart chart = ChartFactory.createXYLineChart("", xLabel, yLabel, dataset);
		XYPlot plot = chart.getXYPlot();
		NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		Font font = new Font("CourierNew", 0, 13);
		plot.getDomainAxis().setLabelFont(font);
		axis.setLabelFont(font);

		plot.setRenderer(new XYLineAndShapeRenderer());

		show(chart);
	}

	public void line(String xLabel, String yLabel, double[] xData, double[] yData) {
		XYSeries series = new XYSeries("");
		int len = xData.length;
		for (int i = 0; i < len; i++)
			series.add(xData[i], yData[i]);

		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(series);
		JFreeChart chart = ChartFactory.createXYLineChart("", xLabel, yLabel, dataset);
		XYPlot plot = chart.getXYPlot();
		NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		Font font = new Font("CourierNew", 0, 13);
		plot.getDomainAxis().setLabelFont(font);
		axis.setLabelFont(font);

		plot.setRenderer(new XYLineAndShapeRenderer());

		show(chart);
	}

	public void lines(String xLabel, String yLabel, List<String> seriesLabel, double[] xData, double[][] yData) {
		int len = xData.length;
		int dim = yData[0].length;
		XYSeriesCollection dataset = new XYSeriesCollection();
		for (int k = 0; k < dim; k++) {
			XYSeries data = new XYSeries(seriesLabel.get(k));
			for (int i = 0; i < len; i++)
				data.add(xData[i], yData[i][k]);
			dataset.addSeries(data);
		}

		JFreeChart chart = ChartFactory.createXYLineChart("", xLabel, yLabel, dataset);
		XYPlot plot = chart.getXYPlot();
		NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		Font font = new Font("CourierNew", 0, 13);
		plot.getDomainAxis().setLabelFont(font);
		axis.setLabelFont(font);

		show(chart);
	}

	/**
	 * Combined lines with the same domain
	 * 
	 * @param xLabel
	 * @param seriesLabel
	 * @param xData
	 * @param yData
	 */
	public void combinedLines(String xLabel, List<String> seriesLabel, double[] xData, double[][] yData) {
		int len = xData.length;
		int numCAT = yData[0].length;

		CombinedDomainXYPlot plot = new CombinedDomainXYPlot(new NumberAxis(xLabel));

		for (int k = 0; k < numCAT; k++) {
			XYPlot subplot = new XYPlot();
			XYSeries series = new XYSeries(seriesLabel.get(k));
			for (int i = 0; i < len; i++)
				series.add(xData[i], yData[i][k]);
			XYSeriesCollection dataset = new XYSeriesCollection();

			dataset.addSeries(series);
			subplot.setDataset(dataset);
			subplot.setRenderer(new XYLineAndShapeRenderer());

			NumberAxis rangeAxis = new NumberAxis("");
			rangeAxis.setAutoRangeIncludesZero(false);
			subplot.setRangeAxis(rangeAxis);

			plot.add(subplot);
		}

		JFreeChart chart = new JFreeChart("", plot);
		// chart.getLegend().setVisible(false);
		show(chart);
	}

	public void combinedLines(String xLabel, List<String> seriesLabel, double[] xData, List<double[][]> yDataList) {
		int len = xData.length;

		CombinedDomainXYPlot plot = new CombinedDomainXYPlot(new NumberAxis(xLabel));

		// LegendItemCollection legend = null;
		for (int t = 0; t < yDataList.size(); t++) {
			double[][] yData = yDataList.get(t);
			int numCAT = yData[0].length;

			XYPlot subplot = new XYPlot();
			XYSeriesCollection dataset = new XYSeriesCollection();

			for (int k = 0; k < numCAT; k++) {
				XYSeries series = new XYSeries(seriesLabel.get(k));
				for (int i = 0; i < len; i++)
					series.add(xData[i], yData[i][k]);
				dataset.addSeries(series);
			}
			subplot.setDataset(dataset);
			subplot.setRenderer(new XYLineAndShapeRenderer());

			NumberAxis rangeAxis = new NumberAxis("");
			rangeAxis.setAutoRangeIncludesZero(false);
			subplot.setRangeAxis(rangeAxis);

			// if (t == 0)
			// legend = subplot.getLegendItems();
			//
			// subplot.setFixedLegendItems(null);
			plot.add(subplot);
		}

		// plot.setFixedLegendItems(legend);
		JFreeChart chart = new JFreeChart("", plot);
		// chart.getLegend().setVisible(false);
		show(chart);
	}

	public void dueLines(String xLabel, List<String> seriesLabel, double[] xData, double[][] yData) {

		int len = xData.length;
		int numCAT = yData[0].length;

		XYPlot plot = new XYPlot();
		plot.setDomainAxis(new NumberAxis(xLabel));

		for (int k = 0; k < numCAT; k++) {
			XYSeries series = new XYSeries(seriesLabel.get(k));
			for (int i = 0; i < len; i++)
				series.add(xData[i], yData[i][k]);
			XYSeriesCollection dataset = new XYSeriesCollection();

			dataset.addSeries(series);
			plot.setDataset(k, dataset);
			plot.setRenderer(k, new XYLineAndShapeRenderer());

			NumberAxis rangeAxis = new NumberAxis(seriesLabel.get(k));
			rangeAxis.setAutoRangeIncludesZero(false);
			plot.setRangeAxis(k, rangeAxis);
			plot.mapDatasetToRangeAxis(k, k);
		}

		JFreeChart chart = new JFreeChart("", plot);
		// chart.getLegend().setVisible(false);
		show(chart);
	}

	public void scatterline(String xLabel, String yLabel, List<String> seriesLabel, List<String> labelList,
			double[][] data, double[] xData, double[][] yData) {
		int len = xData.length;
		int numCAT = yData[0].length;

		XYPlot plot = new XYPlot();
		plot.setDomainAxis(new NumberAxis(xLabel));

		for (int k = 0; k < numCAT; k++) {
			XYSeries series = new XYSeries(seriesLabel.get(k));
			for (int i = 0; i < len; i++)
				series.add(xData[i], yData[i][k]);
			XYSeriesCollection dataset = new XYSeriesCollection();

			dataset.addSeries(series);
			plot.setDataset(k, dataset);
			plot.setRenderer(k, new XYLineAndShapeRenderer());

			NumberAxis rangeAxis = new NumberAxis(seriesLabel.get(k));
			rangeAxis.setAutoRangeIncludesZero(true);
			plot.setRangeAxis(k, rangeAxis);
			plot.mapDatasetToRangeAxis(k, k);
		}

		XYSeriesCollection dataset = new XYSeriesCollection();

		Map<String, List<Integer>> clustering = IntStream.range(0, data.length).boxed()
				.collect(Collectors.groupingBy(i -> labelList.get(i)));

		for (String cluster : clustering.keySet()) {
			XYSeries series = new XYSeries(cluster);
			for (int i : clustering.get(cluster))
				series.add(data[i][0], data[i][1]);
			dataset.addSeries(series);
		}

		int radius = 8;
		Shape shape = new Ellipse2D.Double(-radius / 2, -radius / 2, radius, radius);
		XYShapeRenderer renderer = new XYShapeRenderer();
		renderer.setBaseShape(shape);

		plot.setDataset(numCAT, dataset);
		plot.setRenderer(numCAT, renderer);

		JFreeChart chart = new JFreeChart("", plot);
		show(chart);
	}

	public void scatter(String xLabel, String yLabel, double[][] data) {
		int len = data.length;

		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries series = new XYSeries("");
		for (int i = 0; i < len; i++)
			series.add(data[i][0], data[i][1]);
		dataset.addSeries(series);

		JFreeChart chart = ChartFactory.createScatterPlot("", xLabel, yLabel, dataset);

		XYPlot xyPlot = (XYPlot) chart.getPlot();
		xyPlot.setDomainCrosshairVisible(true);
		xyPlot.setRangeCrosshairVisible(true);

		Shape cross = ShapeUtilities.createDiagonalCross(1, 1);
		XYItemRenderer renderer = xyPlot.getRenderer();
		renderer.setSeriesShape(0, cross);
		renderer.setSeriesPaint(0, Color.blue);

		NumberAxis axis = (NumberAxis) xyPlot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		show(chart);
	}

	public void scatter(String xLabel, String yLabel, List<double[][]> dataList) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		for (int k = 0; k < dataList.size(); k++) {
			double[][] data = dataList.get(k);
			int len = data.length;
			XYSeries series = new XYSeries(k);
			for (int i = 0; i < len; i++)
				series.add(data[i][0], data[i][1]);
			dataset.addSeries(series);
		}

		JFreeChart chart = ChartFactory.createScatterPlot("", xLabel, yLabel, dataset);
		XYPlot plot = chart.getXYPlot();
		NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		show(chart);
	}

	/**
	 * Scatter Plot
	 * 
	 * @param labelList
	 *            classification labels of points
	 * @param dataList
	 */
	public void scatter(List<String> labelList, List<double[][]> dataList) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		for (int k = 0; k < dataList.size(); k++) {
			double[][] data = dataList.get(k);
			int len = data.length;
			XYSeries series = new XYSeries(labelList.get(k));
			for (int i = 0; i < len; i++)
				series.add(data[i][0], data[i][1]);
			dataset.addSeries(series);
		}

		XYPlot plot = new XYPlot();
		plot.setDataset(dataset);
		NumberAxis axis = (NumberAxis) plot.getRangeAxis();
		axis.setAutoRange(true);
		axis.setAutoRangeIncludesZero(false);

		JFreeChart chart = new JFreeChart("", plot);
		show(chart);
	}

	/**
	 * Scatter Plot
	 * 
	 * @param xLabel
	 * @param yLabel
	 * @param labelList
	 *            classification labels of points
	 * @param dataList
	 */
	public void scatter(String xLabel, String yLabel, List<String> labelList, List<double[][]> dataList) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		for (int k = 0; k < dataList.size(); k++) {
			double[][] data = dataList.get(k);
			int len = data.length;
			XYSeries series = new XYSeries(labelList.get(k));
			for (int i = 0; i < len; i++)
				series.add(data[i][0], data[i][1]);
			dataset.addSeries(series);
		}

		NumberAxis xAxis = new NumberAxis(xLabel);
		NumberAxis yAxis = new NumberAxis(yLabel);
		xAxis.setAutoRange(true);
		xAxis.setAutoRangeIncludesZero(false);
		yAxis.setAutoRange(true);
		yAxis.setAutoRangeIncludesZero(false);

		int radius = 8;
		Shape shape = new Ellipse2D.Double(-radius / 2, -radius / 2, radius, radius);
		XYShapeRenderer renderer = new XYShapeRenderer();
		renderer.setBaseShape(shape);

		XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);

		JFreeChart chart = new JFreeChart("", plot);
		show(chart);
	}

	public void scatter(String xLabel, String yLabel, List<String> labelList, double[][] data) {
		XYSeriesCollection dataset = new XYSeriesCollection();

		Map<String, List<Integer>> clustering = IntStream.range(0, data.length).boxed()
				.collect(Collectors.groupingBy(i -> labelList.get(i)));

		for (String cluster : clustering.keySet()) {
			XYSeries series = new XYSeries(cluster);
			for (int i : clustering.get(cluster))
				series.add(data[i][0], data[i][1]);
			dataset.addSeries(series);
		}

		NumberAxis xAxis = new NumberAxis(xLabel);
		NumberAxis yAxis = new NumberAxis(yLabel);
		xAxis.setAutoRange(true);
		xAxis.setAutoRangeIncludesZero(false);
		yAxis.setAutoRange(true);
		yAxis.setAutoRangeIncludesZero(false);

		int radius = 8;
		Shape shape = new Ellipse2D.Double(-radius / 2, -radius / 2, radius, radius);
		XYShapeRenderer renderer = new XYShapeRenderer();
		renderer.setBaseShape(shape);

		XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);

		JFreeChart chart = new JFreeChart("", plot);
		show(chart);
	}

	public void stackedBar3D(String xLabel, String yLabel, double[][] data) {
		CategoryDataset dataset = DatasetUtilities.createCategoryDataset("Group ", "", data);
		JFreeChart chart = ChartFactory.createStackedBarChart3D("", xLabel, yLabel, dataset, PlotOrientation.VERTICAL,
				true, true, false);
		show(chart);
	}

	public void heatmap(double[][] matrix) {
		heatmap("x", "y", matrix);
	}

	public void heatmap(String xTickTitle, String yTickTitle, double[][] matrix) {
		int numRow = matrix.length, numCol = matrix[0].length;
		List<String> xTickLabelList = new ArrayList<>(), yTickLabelList = new ArrayList<>();
		for (int i = 0; i < numRow; i++)
			xTickLabelList.add("" + (i + 1));

		for (int j = 0; j < numCol; j++)
			yTickLabelList.add("" + (j + 1));
		heatmap(xTickTitle, yTickTitle, xTickLabelList, yTickLabelList, matrix);
	}

	public void heatmap(String xTickTitle, String yTickTitle, List<String> xTickLabelList, List<String> yTickLabelList,
			double[][] matrix) {
		final int szColorMap = 12;

		DefaultXYZDataset dataset = new DefaultXYZDataset();
		int numRow = matrix.length, numCol = matrix[0].length;
		final double[][] seriesData = new double[3][numRow * numCol];
		double minimum = Double.MAX_VALUE, maximum = Double.MIN_VALUE;
		for (int j = 0; j < numCol; j++) {
			for (int i = 0; i < numRow; i++) {
				int z = numRow * j + i;
				seriesData[0][z] = i;
				seriesData[1][z] = j;
				seriesData[2][z] = matrix[i][j];
				if (matrix[i][j] < minimum)
					minimum = matrix[i][j];
				else if (matrix[i][j] > maximum)
					maximum = matrix[i][j];
			}
			dataset.addSeries("", seriesData);
		}

		final PlotOrientation plotOrientation = PlotOrientation.VERTICAL;
		final SymbolAxis xAxis = new SymbolAxis(xTickTitle, xTickLabelList.toArray(new String[0]));
		xAxis.setVisible(true);
		// xAxis.setAxisLineVisible(true);
		// xAxis.setTickMarksVisible(true);
		// xAxis.setTickLabelInsets(new RectangleInsets(0.0, 0.0, 0.0, 0.0));
		// xAxis.setTickLabelPaint(Color.BLUE);
		// xAxis.setTickLabelFont(xAxis.getTickLabelFont().deriveFont(8.0f));
		// xAxis.setVerticalTickLabels(true);
		// xAxis.setLowerMargin(10.0);
		// xAxis.setUpperMargin(0.0);

		final SymbolAxis yAxis = new SymbolAxis(yTickTitle, yTickLabelList.toArray(new String[0]));
		yAxis.setVisible(true);
		yAxis.setInverted(true);
		// yAxis.setAxisLineVisible(true);
		// yAxis.setTickMarksVisible(true);
		// yAxis.setTickLabelInsets(new RectangleInsets(0.0, 0.0, 0.0, 0.0));
		// yAxis.setTickLabelPaint(Color.BLUE);
		// yAxis.setTickLabelFont(yAxis.getTickLabelFont().deriveFont(8.0f));
		// yAxis.setVerticalTickLabels(false);
		// yAxis.setLowerMargin(10.0);
		// yAxis.setUpperMargin(0.0);

		final XYBlockRenderer renderer = new XYBlockRenderer();
		renderer.setBaseCreateEntities(true);
		LookupPaintScale paintScale = new LookupPaintScale(minimum, maximum, Color.lightGray);
		double step = (maximum - minimum) / szColorMap;
		for (int i = 0; i < szColorMap; i++)
			paintScale.add(minimum + i * step, RainBow.getColor(i));
		renderer.setPaintScale(paintScale);

		final XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
		plot.setDomainAxis(xAxis);
		plot.setDomainAxisLocation(AxisLocation.TOP_OR_LEFT);
		// plot.setDomainGridlinesVisible(false);

		plot.setRangeAxisLocation(AxisLocation.TOP_OR_LEFT);
		// plot.setRangeGridlinesVisible(false);

		// plot.setOutlineVisible(false);
		plot.setInsets(new RectangleInsets(1.0, 1.0, 1.0, 1.0));
		plot.setAxisOffset(new RectangleInsets(1.0, 1.0, 1.0, 1.0));
		plot.setOrientation(plotOrientation);

		final JFreeChart chart = new JFreeChart("", plot);
		chart.removeLegend();
		chart.setBorderVisible(true);
		chart.setPadding(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
		chart.setAntiAlias(false);
		chart.setTextAntiAlias(true);

		NumberAxis scaleAxis = new NumberAxis("");
		PaintScaleLegend legend = new PaintScaleLegend(paintScale, scaleAxis);
		legend.setSubdivisionCount(128);
		legend.setAxisLocation(AxisLocation.TOP_OR_RIGHT);
		legend.setPadding(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
		legend.setStripWidth(15);
		legend.setPosition(RectangleEdge.RIGHT);
		legend.setBackgroundPaint(chart.getBackgroundPaint());

		chart.addSubtitle(legend);

		show(chart);
	}

	/**
	 * Plot charts in grid
	 * 
	 * @param numRow
	 * @param numCol
	 * @param chartList
	 */
	public void grid(int numRow, int numCol, List<JFreeChart> chartList) {
		JPanel contentPane = new JPanel();
		contentPane.setLayout(new GridLayout(numRow, numCol));

		for (JFreeChart chart : chartList)
			contentPane.add(new ChartPanel(chart));

		frame.setContentPane(contentPane);
		frame.setSize(new Dimension(width, height));
		frame.pack();
		RefineryUtilities.centerFrameOnScreen(frame);
		frame.setVisible(true);
	}

	/**
	 * Plot charts in grid
	 * 
	 * @param xLabel
	 * @param seriesLabel
	 * @param xData
	 * @param yDataList
	 * @param numRow
	 * @param numCol
	 */
	public void grid(String xLabel, List<String> seriesLabel, double[] xData, List<double[][]> yDataList, int numRow,
			int numCol) {
		int len = xData.length;
		int size = yDataList.size();

		List<JFreeChart> chartList = new ArrayList<>();
		for (int t = 0; t < size; t++) {
			double[][] yData = yDataList.get(t);
			int numCAT = yData[0].length;

			XYPlot plot = new XYPlot();
			XYSeriesCollection dataset = new XYSeriesCollection();

			for (int k = 0; k < numCAT; k++) {
				XYSeries series = new XYSeries(seriesLabel.get(k));
				for (int i = 0; i < len; i++)
					series.add(xData[i], yData[i][k]);
				dataset.addSeries(series);
			}
			plot.setDataset(dataset);
			plot.setRenderer(new XYLineAndShapeRenderer());

			NumberAxis rangeAxis = new NumberAxis("");
			rangeAxis.setAutoRangeIncludesZero(false);
			plot.setRangeAxis(rangeAxis);

			NumberAxis domainAxis = new NumberAxis("");
			domainAxis.setAutoRangeIncludesZero(false);
			plot.setDomainAxis(domainAxis);

			chartList.add(new JFreeChart("", plot));
		}
		grid(numRow, numCol, chartList);
	}

	public static void main1212(String[] args) {
		TickClock.beginTick();

		Ace ace = new Ace("");
		int nSample = 1000;

		double[] weights = { 2, 3, -1 };
		BiPredicate<Double, Double> predicate = (x1, x2) -> weights[0] * x1 + weights[1] * x2 + weights[2] > 0;

		List<String> labelList = new ArrayList<>();
		double[][] data = new double[nSample][2];
		double[] x = new double[nSample];
		double[][] y = new double[nSample][1];

		for (int i = 0; i < nSample; i++) {
			double x1 = MathLib.Rand.uniform(-5, 5);
			double x2 = MathLib.Rand.uniform(-5, 5);
			if (predicate.test(x1, x2))
				labelList.add("+1");
			else
				labelList.add("-1");
			data[i][0] = x1;
			data[i][1] = x2;

			x[i] = -5 + i * 10.0 / nSample;
			// 2x1 + 3x2 - 1 = 0
			y[i][0] = 1.0 / 3 - 2.0 / 3 * x[i];
		}

		ace.scatterline("x1", "x2", Arrays.asList("f"), labelList, data, x, y);

		TickClock.stopTick();
	}
}
