package edu.kaist.uilab.NoSyu.examples;

import edu.kaist.uilab.NoSyu.LDA.DistributedOnline.LDA_DOnline_Driver;

public class Distributed_Online_LDA_Example 
{

	public static void main(String[] args) 
	{
		// Just run LDA_DOnline_Driver main function
		try
		{
			LDA_DOnline_Driver.main(args);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}

}
