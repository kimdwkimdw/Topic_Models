package edu.kaist.uilab.NoSyu.LDA.Online;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Date;

import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class Online_LDA_Example 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int minibatch_size;
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
	public static void main(String[] args) 
	{
		try
		{
			TopicNum = Integer.parseInt(args[0]);
			Max_Iter = Integer.parseInt(args[1]);
			minibatch_size = Integer.parseInt(args[2]);
			voca_file_path = new String(args[3]);
			BOW_file_path = new String(args[4]);
			output_file_name = new String(args[5]);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: TopicNum Max_Iter Minibatch_size voca_file_path BOW_file_path output_file_name");
			System.exit(1);
		}
		
		Run_Online_LDA();
	}

	/*
	 * Run Online LDA
	 * */
	private static void Run_Online_LDA()
	{
		ArrayList<String> wordList = null;
		ArrayList<Document_LDA_Online> documents = null;
		
		try
		{
			wordList = Miscellaneous_function.makeStringListFromFile(voca_file_path);
			documents = make_document_list();
			Miscellaneous_function.Print_String_with_Date("Finish making documents list");
			
			Miscellaneous_function.Open_Target_File("Print_String_with_Date_oLDA.txt");
		}
		catch(java.lang.Throwable t)
		{
			t.printStackTrace();
			System.exit(1);
		}
		
		long lStartTime = new Date().getTime(); //start time
		
		Online_LDA oLDAs = new Online_LDA(TopicNum, Max_Iter, minibatch_size, wordList, documents);
		
		Miscellaneous_function.Print_String_with_Date("Online LDA Starts!");
		oLDAs.oLDA_run();
		oLDAs.ExportResultCSV(output_file_name);
		
		long lEndTime = new Date().getTime(); //end time
		
		long difference = lEndTime - lStartTime; //check different
		
		Miscellaneous_function.Print_String_with_Date("Elapsed milliseconds: " + difference);
		
		Miscellaneous_function.Close_Target_File();
	}
	
	/*
	 * Make Documents list
	 * */
	private static ArrayList<Document_LDA_Online> make_document_list()
	{
		ArrayList<Document_LDA_Online> documents = new ArrayList<Document_LDA_Online>();
		
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(BOW_file_path)));
			String line = null;
			while((line=in.readLine()) != null)
			{
				Document_LDA_Online doc = new Document_LDA_Online(line);

				documents.add(doc);
			}
			in.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in make_document_list function in oLDA_Main class");
			t.printStackTrace();
			System.exit(1);
		}
		
		return documents;
	}
	
}
