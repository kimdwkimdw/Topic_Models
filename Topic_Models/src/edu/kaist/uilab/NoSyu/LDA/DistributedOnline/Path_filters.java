package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;

public class Path_filters 
{
	public static class Lambda_Filter implements PathFilter 
	{
		public boolean accept(Path path) 
		{
			return path.toString().contains("lambda");
		}
	}
	
	
	public static class Alpha_Filter implements PathFilter 
	{
		public boolean accept(Path path) 
		{
			return path.toString().contains("alpha");
		}
	}
}
