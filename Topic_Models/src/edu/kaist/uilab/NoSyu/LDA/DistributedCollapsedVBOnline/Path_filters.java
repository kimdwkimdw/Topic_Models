package edu.kaist.uilab.NoSyu.LDA.DistributedCollapsedVBOnline;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;

public class Path_filters 
{
	public static class sum_phi_dvk_d_E_Filter implements PathFilter 
	{
		public boolean accept(Path path) 
		{
			return path.toString().contains("sum_phi_dvk_d_E");
		}
	}
	
	public static class sum_phi_dvk_dv_E_Filter implements PathFilter 
	{
		public boolean accept(Path path) 
		{
			return path.toString().contains("sum_phi_dvk_dv_E");
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
