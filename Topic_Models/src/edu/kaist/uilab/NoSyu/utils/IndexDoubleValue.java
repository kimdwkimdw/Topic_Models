package edu.kaist.uilab.NoSyu.utils;

import java.util.Comparator;

public class IndexDoubleValue
{
	static public class IndexDoubleValueComparator implements Comparator<IndexDoubleValue>
	{
		// Desending Order
	    @Override
	    public int compare(IndexDoubleValue x, IndexDoubleValue y)
	    {
	        if (x.value > y.value)
	        {
	            return -1;
	        }
	        else if (x.value < y.value)
	        {
	            return 1;
	        }
	        else
	        {
	        	// 같다.
	        	if(x.index < y.index)
	        	{
	        		return -1;
	        	}
	        	else
	        	{
	        		return 1;
	        	}
	        }
	    }
	}
	
	public int index;
	public double value;
	
	IndexDoubleValue()
	{
		this.index = 0;
		this.value = 0;
	}
	
	IndexDoubleValue(int idx, double v)
	{
		this.index = idx;
		this.value = v;
	}
}


