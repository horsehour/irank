package com.horsehour.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

/**
 * configuration for training algorithm
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140304
 */
public class Messenger {
	private Properties properties;

	public Messenger() {
		properties = new Properties();
	}

	public Messenger(String key, String value) {
		this();
		properties.setProperty(key, value);
	}

	public Messenger(String propFile) throws FileNotFoundException, IOException {
		this();

		File file = new File(propFile);
		properties.load(new FileInputStream(file));
	}

	public String get(String key) {
		return properties.getProperty(key);
	}

	public void set(String key, String value) {
		properties.setProperty(key, value);
	}

	public void setNumOfIter(int n) {
		properties.setProperty("nIter", n + "");
	}

	public int getNumOfIter() {
		String val = properties.getProperty("nIter");
		return Integer.parseInt(val);
	}
}
