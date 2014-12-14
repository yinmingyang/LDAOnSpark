LDAOnSpark
==========
This is a high efficiency distributed Gibbs Sampling inference for Latent Dirichlet Allocation(LDA). This algorithm uses Spark and Breeze. You need to setup Scala and Spark before using this code. An experimental set with 5000 documents could be trained in 5 minutes on a cluster of 30 cores. To use this code, you need to pre-process your documents. The input format should be an RDD of class Data:  
**inputData = RDD[Data]**  
  
And Data is:  
**class Data(ID: String, word: Array[Int])**  
  
ID is the document ID. Word is an array of integer, which represents the index of words in this document.  
  
Then you can create an LDA class and set the parameters like the number of topics, number of iteration and so on. **Remember that vocabulary must be set. The vocabulary should be a file that each line is a single word.** The line number is the index of this word. After that, you can use LDA.run(inputData) for training. The result is a Result class, you can get the matrixes alpha and beta by function getAlpha and getBeta.  
  
The algorithm is based on "Distributed Inference for Latent Dirichlet Allocation", available at
 [http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2007_672.pdf]
For more information about LDA, please refer to "Latent Dirichlet Allocation", available at
 [http://dl.acm.org/citation.cfm?id=944937]
