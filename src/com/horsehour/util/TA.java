package com.horsehour.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:33:57 PM, Mar 2, 2017
 *
 */

public class TA {
	static OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

	public static void main(String[] args) throws IOException {
		String base = "/Users/chjiang/Downloads/";
		Path output = Paths.get("/Users/chjiang/Downloads/all_grades.csv");
		List<String> idList = Files.readAllLines(Paths.get(base + "inclusive.csv"));

		List<String> lines;
		for (String id : idList) {
			Path file = Paths.get(base + "/all_grades/" + id + "_summary.json");
			lines = Files.readAllLines(file);
			analyze(id, lines, output);
		}
	}

	public static void analyze(String id, List<String> lines, Path output) throws IOException {
		StringBuffer sb = new StringBuffer();
		int g = 0, i = 0, count = 0;
		for (; i < lines.size(); i++) {
			String line = lines.get(i);
			if (g == 0 && line.contains("user_id")) {
				int ind = line.indexOf(":");
				line = line.substring(ind + 1).replaceAll("\"", "");
				line = line.replace(",", "").trim();
				if (!id.equals(line)) {
					System.err.println("Analyzed id is inconsistent with given id.");
					continue;
				}

				sb.append(line + "\t");
				g = 1;
			} else if (g == 1 && line.contains("\"score\"")) {
				int ind = line.indexOf(":");
				line = line.substring(ind + 1).replaceAll("\"", "");
				line = line.replace(",", "").trim();
				sb.append(line + "\t");
				g = 2;
				count++;
			} else if (g == 2 && line.contains("\"days_late\"")) {
				int ind = line.indexOf(":");
				line = line.substring(ind + 1).replaceAll("\"", "");
				line = line.replace(",", "").trim();
				sb.append(line);
				g = 1;

				if (count == 4) {
					sb.append("\r\n");
					break;
				} else
					sb.append("\t");
			}
		}
		Files.write(output, sb.toString().getBytes(), options);
	}
}
