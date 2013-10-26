package edu.kaist.uilab.NoSyu.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import edu.kaist.uilab.NoSyu.LDA.Gibbs.AD_LDA_Gibbs;
import edu.kaist.uilab.NoSyu.LDA.Gibbs.Document_LDA_Gibbs;
import edu.kaist.uilab.NoSyu.utils.*;

public class AD_LDA_Gibbs_Example 
{
	private static int numSampling;
	private static int thread_num = 8;
	private static int numTopic = 500;
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_path = null;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception
	{
		try
		{
			thread_num = Integer.parseInt(args[0]);
			numTopic = Integer.parseInt(args[1]);
			numSampling = Integer.parseInt(args[2]);
			voca_file_path = new String(args[3]);
			BOW_file_path = new String(args[4]);
			output_file_path = new String(args[5]);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: thread_num numTopic numSampling voca_file_path BOW_file_path output_file_path");
			System.exit(1);
		}
		
		SimpleADLDATest();
	}
	
	/*
	 * 
	 * */
	private static void SimpleADLDATest() throws Exception
	{
		List<String> wordList = Miscellaneous_function.makeStringListFromFile(voca_file_path);
		List<Document_LDA_Gibbs> documents = generateDocumentListForLDA();
		System.out.println("Finish making documents list");
		
		Miscellaneous_function.Open_Target_File("Print_String_with_Date_ADLDA.txt");
		
		long lStartTime = new Date().getTime(); //start time
		
		AD_LDA_Gibbs ADLDAs = new AD_LDA_Gibbs(numTopic, wordList, documents, thread_num);
		System.out.println("AD_LDA Starts!");
		ADLDAs.run(numSampling, voca_file_path, output_file_path);
		ADLDAs.ExportResultCSV(output_file_path);
		
		long lEndTime = new Date().getTime(); //end time
		
		long difference = lEndTime - lStartTime; //check different
		
		Miscellaneous_function.Print_String_with_Date("Elapsed milliseconds: " + difference);
		
		Miscellaneous_function.Close_Target_File();
	}
	
	private static List<Document_LDA_Gibbs> generateDocumentListForLDA() throws IOException 
	{
		ArrayList<Document_LDA_Gibbs> documents = new ArrayList<Document_LDA_Gibbs>();
		
		BufferedReader in = new BufferedReader(new FileReader(new File(BOW_file_path)));
		String line = null;
		int doc_idx = 0;
		
		try
		{
			while((line = in.readLine()) != null)
			{
				if(line.length() > 1)
				{
					Document_LDA_Gibbs doc = new Document_LDA_Gibbs(doc_idx, line);
					
					documents.add(doc);
					doc_idx++;
				}
			}
		}
		catch(java.lang.Throwable t)
		{
			t.printStackTrace();
			System.err.println(doc_idx);
			System.err.println(line);
			System.exit(1);
		}
		
		in.close();
		
		return documents;
	}
}
