package com.horsehour.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:33:57 PM, Mar 2, 2017
 */

public class TA {
	static OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

	public static void main(String[] args) throws IOException {
		String base = "/Users/chjiang/GitHub/courses/ai/";
		Path output = Paths.get(base + "final.csv");
		List<Path> files = Files.list(Paths.get(base + "/grading/raw_data/")).collect(Collectors.toList());
		List<String> lines;
		for (Path file : files) {
			String name = file.toFile().getName();
			String id = name.substring(0, name.indexOf("_"));
			lines = Files.readAllLines(file);
			analyze(id, lines, output);
		}

		// for (String id : idList) {
		// Path file = Paths.get(base + "/grading/raw_data/" + id +
		// "_summary.json");
		// lines = Files.readAllLines(file);
		// analyze(id, lines, output);
		// }
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
			}else if(g == 1 && line.contains("\"registration_section\": 0")){
				return;
			}else if (g == 1 && line.contains("\"score\"")) {
				int ind = line.indexOf(":");
				line = line.substring(ind + 1).replaceAll("\"", "");
				line = line.replace(",", "").trim();
				sb.append(line + "\t");
				count++;
				if (count > 2)
					g = 2;
			} else if (g == 2 && line.contains("\"days_late\"")) {
				int ind = line.indexOf(":");
				line = line.substring(ind + 1).replaceAll("\"", "");
				line = line.replace(",", "").trim();
				sb.append(line);
				g = 1;

				if (count == 10) {
					sb.append("\r\n");
					break;
				} else
					sb.append("\t");
			}
		}
		Files.write(output, sb.toString().getBytes(), options);
	}
}
