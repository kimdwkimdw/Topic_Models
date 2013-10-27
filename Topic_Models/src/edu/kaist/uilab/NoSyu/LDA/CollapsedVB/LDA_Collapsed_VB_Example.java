package edu.kaist.uilab.NoSyu.LDA.CollapsedVB;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;


public class LDA_Collapsed_VB_Example 
{
	private static int numTopic = 500;
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_path = null;
	private static int numSampling = 2000;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception
	{
		try
		{
			numTopic = Integer.parseInt(args[0]);
			numSampling = Integer.parseInt(args[1]);
			voca_file_path = new String(args[2]);
			BOW_file_path = new String(args[3]);
			output_file_path = new String(args[4]);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: numTopic numSampling voca_file_path BOW_file_path output_file_path");
			System.exit(1);
		}
		
		SimpleCVBTest();
	}
	
	/*
	 * 
	 * */
	private static void SimpleCVBTest() throws Exception
	{
		List<String> wordList = Miscellaneous_function.makeStringListFromFile(voca_file_path);
		ArrayList<Document_LDA_CollapsedVB> documents = null;
		documents = generateDocumentListForLDA(BOW_file_path);
		System.out.println("Finish making documents list");
		
		Miscellaneous_function.Open_Target_File("Print_String_with_Date_LDA_CollapsedVB.txt");
		
		long lStartTime = new Date().getTime(); //start time
		LDA_Collapsed_VB LDACoVI = new LDA_Collapsed_VB(numTopic, wordList, documents);
		System.out.println("LDA_Collapsed_VB Starts!");
		LDACoVI.run(numSampling, voca_file_path);
		LDACoVI.ExportResultCSV(output_file_path);
		
		long lEndTime = new Date().getTime(); //end time
		
		long difference = lEndTime - lStartTime; //check different
		
		Miscellaneous_function.Print_String_with_Date("Elapsed milliseconds: " + difference);
		
		Miscellaneous_function.Close_Target_File();
	}
	
	private static ArrayList<Document_LDA_CollapsedVB> generateDocumentListForLDA(String target_BOW_file_path) throws IOException 
	{
		ArrayList<Document_LDA_CollapsedVB> documents = new ArrayList<Document_LDA_CollapsedVB>();
		
		BufferedReader in = new BufferedReader(new FileReader(new File(target_BOW_file_path)));
		String line = null;

		while((line=in.readLine()) != null)
		{
			Document_LDA_CollapsedVB doc = new Document_LDA_CollapsedVB(numTopic, line);
			
			documents.add(doc);
		}
		in.close();
		
		return documents;
	}
}
