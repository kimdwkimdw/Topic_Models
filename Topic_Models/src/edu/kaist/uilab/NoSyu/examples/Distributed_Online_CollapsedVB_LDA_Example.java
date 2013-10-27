package edu.kaist.uilab.NoSyu.examples;

import edu.kaist.uilab.NoSyu.LDA.DistributedCollapsedVBOnline.DOCLDA_Driver;

public class Distributed_Online_CollapsedVB_LDA_Example 
{

	public static void main(String[] args) 
	{
		// Just run DOCLDA_Driver main function
		try
		{
			DOCLDA_Driver.main(args);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}

}
