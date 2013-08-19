package edu.kaist.uilab.NoSyu.utils;

/*
 * 기본적으로 DongWoo Kim(arongdari@kaist.ac.kr)의 것을 참조하여 작성
 * 
 * */
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

public class IntegerMatrix
{
	private int RowNum;			// 행 크기
	private int ColNum;			// 열 크기
	private int [][] element;	// 행렬에 들어있는 element
	
	/*
	 * 생성자
	 * 
	 * 행 크기, 열 크기
	 * */
	public IntegerMatrix(int RowNum, int ColNum) throws IndexOutOfBoundsException
	{
		if(RowNum < 1 || ColNum < 1)
		{
			throw new IndexOutOfBoundsException();
		}
		
		this.RowNum = RowNum;
		this.ColNum = ColNum;
		this.element = new int[RowNum][ColNum];
	}
	
	/*
	 * Get/Set Methods
	 * */
	public int getRowNum()
	{
		return this.RowNum;
	}

	public int getColNum()
	{
		return this.ColNum;
	}
	
	public int getValue(int RowIdx, int ColIdx)
	{
		return this.element[RowIdx][ColIdx];
	}
	
	/*
	 * Set value
	 * */
	public void setValuetoElement(int RowIdx, int ColIdx, int value)
	{
		this.element[RowIdx][ColIdx] = value;
	}
	
	/*
	 * Set value to all same
	 * */
	public void setValuetoAllElement(int value)
	{
		for(int[] row : this.element)
		{
			Arrays.fill(row, value);
		}
	}
	
	/*
	 * element value를 늘이고 줄이기
	 * */
	public synchronized void incValue(int RowIdx, int ColIdx)
	{
		this.element[RowIdx][ColIdx]++;
	}
	
	public synchronized void incValue(int RowIdx, int ColIdx, int value)
	{
		this.element[RowIdx][ColIdx] += value;
	}
	
	public synchronized void decValue(int RowIdx, int ColIdx)
	{
		this.element[RowIdx][ColIdx]--;
	}
	
	public synchronized void decValue(int RowIdx, int ColIdx, int value)
	{
		this.element[RowIdx][ColIdx] -= value;
	}
	
	/*
	 * Row와 Col에 따라 value 뽑기
	 * */
	public int[] getRow(int RowIdx)
	{
		if(RowIdx > this.RowNum)
		{
			return null;
		}
		return this.element[RowIdx];
	}
	
	public int getRowSum(int RowIdx)
	{
		if(RowIdx > this.RowNum)
		{
			return 0;
		}
		
		int sum = 0;
		for (int idx = 0; idx < this.ColNum; idx++)
		{
			sum += this.element[RowIdx][idx];
		}
		return sum;
	}
	
	public int[] getColumn(int ColIdx)
	{
		if(ColIdx > this.ColNum)
		{
			return null;
		}
		
		int[] col = new int[this.RowNum];
		for(int idx = 0 ; idx < this.RowNum ; idx++)
		{
			col[idx] = this.element[idx][ColIdx];
		}
		return col;
	}
	
	public int getColSum(int ColIdx)
	{
		if(ColIdx > this.ColNum)
		{
			return 0;
		}
		
		int sum = 0;
		for(int idx = 0 ; idx < this.RowNum ; idx++)
		{
			sum += this.element[idx][ColIdx];
		}
		return sum;
	}
	
	public int[][] getArray() 
	{
		return this.element;
	}
	
	/*
	 * Column에 대해 index를 정렬하여 반환한다.
	 * 즉, index - value 쌍에서 value에 맞게 정렬한 결과를 내보낸다.
	 * */
	public Vector<Integer> getSortedColIndex(int col, int n)
	{
		// http://stackoverflow.com/questions/109383/how-to-sort-a-mapkey-value-on-the-values-in-java
		int[] colvec_arr = this.getColumn(col);
		HashMap<Integer, Integer> idx_col_vec = new HashMap<Integer, Integer>();

		// HashMap 채우기
		// idx_col_vec은 index와 그 index에 해당하는 value를 저장
		for(int idx = 0 ; idx < colvec_arr.length ; idx++)
		{
			idx_col_vec.put(idx, colvec_arr[idx]);
		}
		
	    List<Integer> mapKeys = new LinkedList<Integer>(idx_col_vec.keySet());
	    List<Integer> mapValues = new ArrayList<Integer>(idx_col_vec.values());
	    // value를 decending order로 정렬
	    Collections.sort(mapValues);
	    Collections.reverse(mapValues);
	    // 이건 왜 하지?
	    Collections.sort(mapKeys);
	    Vector<Integer> sortedList = new Vector<Integer>();	// value에 따라 정렬된 index가 들어있다.
	    
	    // value를 하나씩 살펴보자
//	    Iterator<Integer> valueIt = mapValues.iterator();
//	    while (valueIt.hasNext())
	    for(int idx = 0 ; idx < n ; idx++)
	    {
//	        Object val = valueIt.next();
//	    	int val = valueIt.next();
	    	int val = mapValues.get(idx);
	        Iterator<Integer> keyIt = mapKeys.iterator();
	        
	        while (keyIt.hasNext())
	        {
//	            Object key = keyIt.next();
	        	Object key_object = keyIt.next();
	            // idx_col_vec에서 key를 통해 값을 뽑아낸다.
//	            Integer comp1 = (Integer)idx_col_vec.get(key);
	        	int comp1 = (Integer)idx_col_vec.get(key_object);
	            // 정렬되어진 value이다.
//	            Integer comp2 = (Integer)val;
	        	int comp2 = val;
	            
	            // 위의 두 개가 같은지 확인한다.
//	            if (comp1.equals(comp2))
	        	if (comp1 == comp2)
	            {
	            	idx_col_vec.remove(key_object);
	                mapKeys.remove(key_object);
	                sortedList.add((Integer)key_object);
	                break;
	            }
	        }
	    }
	    return sortedList;
	}
	
	/*
	 * 결과를 CSV로 출력할 때 사용
	 * */
	public void writeMatrixToCSVFile(String outputFilePath) throws Exception
	{
		PrintWriter out = new PrintWriter(new FileWriter(new File(outputFilePath)));
		
		for(int row = 0; row < this.RowNum ; row++)
		{
			out.print(getValue(row, 0));
			for(int col = 1; col < this.ColNum ; col++)
			{
				out.print("," + this.getValue(row, col));
			}
			out.println();
		}
		
		out.close();
	}

	public void writeMatrixToCSVFile(String outputFilePath, List<String> rowIndexNameList) throws IOException 
	{
		PrintWriter out = new PrintWriter(new FileWriter(new File(outputFilePath)));
		
		for(int row = 0; row < this.RowNum ; row++)
		{
			out.print("\""+ rowIndexNameList.get(row) + "\"");
			
			for(int col=0; col < this.ColNum ; col++)
			{
				out.print(","+getValue(row, col));
			}
			out.println();
		}

		out.close();
		
	}

	public void writeRankingFile(String outputFilePath, List<String> wordList, int numWords) throws IOException 
	{
		ArrayList<Vector<Integer>> index = new ArrayList<Vector<Integer>>();
		PrintWriter out = new PrintWriter(new FileWriter(new File(outputFilePath)));
		// numWords은 기록할 ranking의 개수를 뜻하기도 한다.
		
		// 각 column별로 sorting
		for(int idx = 0 ; idx < this.ColNum ; idx++)
		{
			// 해당 column에서 value에 따른 index가 저장되어진 vector가 index variable에 저장된다.
			index.add(getSortedColIndex(idx, numWords));
			
			out.write("Topic " + idx + ",");
		}
		out.write("\n");
		
		for(int jdx = 0 ; jdx < numWords ; jdx++)
		{
			for(int idx = 0; idx < this.ColNum ; idx++)
			{
				out.write(wordList.get(index.get(idx).get(jdx)) + ",");
			}
			out.write("\n");
		}

		out.close();
	}

	/*
	 * Matrix Transpose
	 * */
	public IntegerMatrix transpose()
	{
		IntegerMatrix X = null;
		
		try
		{
			X = new IntegerMatrix(this.ColNum, this.RowNum);
		}
		catch(IndexOutOfBoundsException IOBE)
		{
			System.err.println("Serious Problem in edu.kaist.uilab.NoSyu.util.IntegerMatrix Class transpose function");
			return null;
		}

		int[][] C = X.getArray();
		for(int idx = 0 ; idx < this.RowNum ; idx++) 
		{
			for(int jdx = 0 ; jdx < this.ColNum ; jdx++) 
			{
				C[jdx][idx] = this.element[idx][jdx];
			}
		}
		return X;
	}
}
