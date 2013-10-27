package edu.kaist.uilab.NoSyu.utils;

import org.apache.hadoop.io.Text;
//import org.apache.hadoop.mapred.lib.MultipleSequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.MultipleTextOutputFormat;

public class Reducer_MultipleOutputFormat extends MultipleTextOutputFormat<Text, Text> 
{
	protected String generateFileNameForKeyValue(Text key, Text value, String name) 
	{
		String keyString = key.toString();
		
		return keyString + "_" + name;
	}
	
	protected Text generateActualKey(Text key, Text value)
	{
		String valueString = value.toString();
		
		String[] valueString_arr = valueString.split("\t");
		
		return new Text(valueString_arr[0]);
	}
	
	protected Text generateActualValue(Text key, Text value)
	{
		String valueString = value.toString();
		
		String[] valueString_arr = valueString.split("\t");
		
		return new Text(valueString_arr[1]);
	}
}
