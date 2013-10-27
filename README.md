Topic_Models
============

Java implementation of Bayesian Topic Models

Main contributor: JinYeong Bak (jy.bak@kaist.ac.kr)

We implement several topic models with various inference methods (Gibbs sampling, Variational Inference, Distributed, Online, etc).

We also upload the technical reports that how to inferece the models. It will be helpful to understand our implementation and topic model itself.


Library

* Apache Commons Math - http://commons.apache.org/proper/commons-math/
* google-gson - https://code.google.com/p/google-gson/
* Hadoop - http://hadoop.apache.org/


Example dataset
* 2246 documents from the Associated Press - http://www.cs.princeton.edu/~blei/lda-c/
 

Example of arguments for each examples
* AD_LDA_Gibbs_Example - 8 100 2000 ./ap_news/vocab.txt ./ap_news/ap.dat ap_news_ADLDA
* Online_LDA_Example - 100 100 64 ./ap_news/vocab.txt ./ap_news/ap.dat ap_news_oLDA
* Distributed_Online_LDA_Example - 100 128 100 ./ap_news/vocab.txt ./ap_news/ap.dat DoLDA_ap_news /user/NoSyu/Distributed_Online_LDA 2 2
* LDA_Collapsed_VB_Example - 100 2000 ./ap_news/vocab.txt ./ap_news/ap.dat ap_news_CVBLDA
* Collapsed_VB_Online_LDA_Example - 100 500 64 ./ap_news/vocab.txt ./ap_news/ap.dat ap_news_SCVBLDA
* Distributed_Online_CollapsedVB_LDA_Example - 100 128 100 ./ap_news/vocab.txt ./ap_news/ap.dat DoLDA_ap_news /user/NoSyu/DOC_LDA 2 2
