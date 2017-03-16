package com.horsehour.ml.cluster;

import java.util.ArrayList;
import java.util.List;

import processing.core.PApplet;
import processing.core.PVector;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 24th Dec. 2015 PM 11:08:52
 **/
public class K_Means extends PApplet {
	public List<PVector> pointList;
	public List<Cluster> clusters;

	public int k = 3;
	public float easing = 0.05f;

	public class Cluster {
		public PVector center;
		public PVector prevCenter;
		public List<PVector> memberList;
		public int clusterColor;

		public Cluster() {
			center = new PVector();
			memberList = new ArrayList<PVector>();
			clusterColor = color(random(255), random(255), random(255));
		}
	}

	@Override
	public void settings(){
		size(600, 600);
		pointList = new ArrayList<PVector>();
		clusters = new ArrayList<Cluster>();
		randSelectCenters(k);
	}

	@Override
	public void mouseClicked(){
		pointList.add(new PVector(mouseX, mouseY));
	}

	@Override
	public void mouseDragged(){
		int r = 20;
		pointList.add(new PVector(mouseX + random(-r, r), mouseY + random(-r, r)));
	}

	@Override
	public void keyPressed(){
		switch (key) {
			case 'p':
				for (int i = 0; i < 20; ++i)
					pointList.add(new PVector(random(width), random(height)));
				return;
			case 'c':
				pointList.clear();
				clusters.clear();
				break;
			case 'r':
				clusters.clear();
				randSelectCenters(k);
				break;
			case 'q':
				exit();
		}

		if (clusters.isEmpty())
			randSelectCenters(k);
		buildMembership();
	}

	@Override
	public void draw(){
		background(0);
		for (PVector v : pointList) {
			fill(255);
			ellipse(v.x, v.y, 5, 5);
		}

		for (Cluster c : clusters) {
			PVector target = c.center;
			c.prevCenter.add(PVector.mult(PVector.sub(target, c.prevCenter), easing));
			fill(c.clusterColor, 127);
			stroke(c.clusterColor, 127);
			for (PVector point : c.memberList)
				line(c.prevCenter.x, c.prevCenter.y, point.x, point.y);
			fill(c.clusterColor);
			ellipse(c.prevCenter.x, c.prevCenter.y, 10, 10);
		}
	}

	/**
	 * select centers randomly
	 * @param k
	 */
	public void randSelectCenters(int k){
		if (pointList.isEmpty())
			return;

		Cluster c = null;
		for (int i = 0; i < k; ++i) {
			c = new Cluster();
			c.center = pointList.get((int) random(0, pointList.size()));
			clusters.add(c);
		}
	}

	/**
	 * Build point's membership based upon its distance to center points
	 */
	public void buildMembership(){
		if (pointList.isEmpty())
			return;

		float distance = 0;
		float minDistance;

		/* clean previous clusters */
		for (Cluster c : clusters)
			c.memberList.clear();

		for (PVector point : pointList) {
			minDistance = width * height * width;
			Cluster closest = clusters.get(0);
			for (Cluster c : clusters) {
				distance = point.dist(c.center);
				if (distance <= minDistance) {
					closest = c;
					minDistance = distance;
				}
			}
			closest.memberList.add(point);
		}

		/* update cluster center */
		for (Cluster c : clusters) {
			c.prevCenter = c.center;

			PVector center = new PVector();// new center
			for (PVector point : c.memberList)
				center.add(point);
			center.div(c.memberList.size());
			c.center = center;
		}
	}

	public void run(){
		String cls = getClass().getName();
		PApplet.main(new String[]{cls});
	}

	public static void main(String[] args){
		K_Means km = new K_Means();
		km.run();
	}
}
