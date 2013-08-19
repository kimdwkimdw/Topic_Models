package edu.kaist.uilab.NoSyu.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.TreeSet;

import org.ejml.simple.SimpleMatrix;

final public class Miscellaneous_function 
{
	private static PrintWriter output_file = null;
	
	public static void Open_Target_File(String filename) throws IOException
	{
		output_file = new PrintWriter(new FileWriter(new File(filename)));
	}
	
	public static void Close_Target_File()
	{
		output_file.close();
	}
	
	/*
	 * Print with Time and date
	 * */
	public static void Print_String_with_Date(String print_str)
	{
		Date date = new Date();
		
		System.out.println(date.toString() + "\t" + print_str);
		
		if(null != output_file)
		{
			output_file.println(date.toString() + "\t" + print_str);
		}
	}
	
	/*
	 * Load file and read it to ArrayList<String> 
	 * */
	public static ArrayList<String> makeStringListFromFile(String file_path) throws Exception 
	{
		ArrayList<String> result = new ArrayList<String>();
		String line = null;
		
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(file_path)));
			
			while((line=in.readLine()) != null)
			{
				result.add(line);
			}
			
			in.close();
		
		}
		catch(FileNotFoundException e)
		{
			System.err.println("File is not existed\t" + file_path);
			e.printStackTrace();
			return null;
		}
		
		return result;
	}
	
	/**
     * This method ensures that the output String has only
     * valid XML unicode characters as specified by the
     * XML 1.0 standard. For reference, please see
     * <a href="http://www.w3.org/TR/2000/REC-xml-20001006#NT-Char">the
     * standard</a>. This method will return an empty
     * String if the input is null or empty.
     *
     * @param in The String whose non-valid characters we want to remove.
     * @return The in String, stripped of non-valid characters.
     */
    public static String stripNonValidXMLCharacters(String in) 
    {
        StringBuffer out = new StringBuffer(); // Used to hold the output.
        char current; // Used to reference the current character.

        if (in == null || ("".equals(in))) return ""; // vacancy test.
        for (int i = 0; i < in.length(); i++) 
        {
            current = in.charAt(i); // NOTE: No IndexOutOfBoundsException caught here; it should not happen.
            if ((current == 0x9) ||
                (current == 0xA) ||
                (current == 0xD) ||
                ((current >= 0x20) && (current <= 0xD7FF)) ||
                ((current >= 0xE000) && (current <= 0xFFFD)) ||
                ((current >= 0x10000) && (current <= 0x10FFFF)))
                out.append(current);
        }
        return out.toString();
    }
    
    
	/*
	 * 
	 * */
    public static int[] Sort_Ranking_Double(double[] input_arr, int max_rank)
	{
		TreeSet<IndexDoubleValue> sorted_set = new TreeSet<IndexDoubleValue>(new IndexDoubleValue.IndexDoubleValueComparator());
		int[] sorted_index = new int[max_rank];
		
		// Sort
		for(int idx = 0 ; idx < input_arr.length ; idx++)
		{
			sorted_set.add(new IndexDoubleValue(idx, input_arr[idx]));
		}
		
		// Get elements that maximum is max_rank
		for(int idx = 0 ; idx < max_rank ; idx++)
		{
			sorted_index[idx] = sorted_set.pollFirst().index;
		}
		
		return sorted_index;
	}
	
	
	/*
	 * 
	 * */
	public static int[] Sort_Ranking_Double(SimpleMatrix input_vec, int max_rank)
	{
		TreeSet<IndexDoubleValue> sorted_set = new TreeSet<IndexDoubleValue>(new IndexDoubleValue.IndexDoubleValueComparator());
		int[] sorted_index = new int[max_rank];
		int num_elements = input_vec.numCols();
		
		// Sort
		for(int idx = 0 ; idx < num_elements ; idx++)
		{
			sorted_set.add(new IndexDoubleValue(idx, input_vec.get(0, idx)));
		}
		
		// Get elements that maximum is max_rank
		for(int idx = 0 ; idx < max_rank ; idx++)
		{
			sorted_index[idx] = sorted_set.pollFirst().index;
		}
		
		return sorted_index;
	}
}
