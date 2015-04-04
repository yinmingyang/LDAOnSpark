import java.io.{FileOutputStream, OutputStreamWriter}

import breeze.collection.mutable.SparseArray
import breeze.linalg.SparseVector
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.util.Random

/*
* Data represents the input format, which contains the document ID, and an array of index of word in
* vocabulary, start from 0.
* */

case class Data(ID: String, word: Array[Int])

/*
*
* LDAModel contains all data needed during iteration.
* ID: document ID
* word: Array of word index
* topic: Sampling result
* gamma: represents for p(Zijp=k)
* count: Local count for each document
*
* n(k) is the document-topic count for each doc
* */

case class LDAModel(ID: String,
                    word: Array[Int],
                    topic: Array[Int],
                    gamma: Array[Array[Double]],
                    count: Array[Int]
                     )

/**
 * Distributed Collapsed Gibbs Sampling algorithm for Latent Dirchlet Allocation(LDA)
 * The algorithm is based on "Distributed Inference for Latent Dirichlet Allocation", available at
 * [http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2007_672.pdf]
 *
 *
 *
 * For more information about LDA, please refer to "Latent Dirichlet Allocation", available at
 * [http://dl.acm.org/citation.cfm?id=944937]
 *
 */


class LDA(var maxIteration: Int,
          var numOfTopic: Int,
          var alpha: Double,
          var beta: Double,
          var vocabulary: Array[String],
          var numOfVoc: Int) extends Serializable {

  def this() = this(50, 10, 0.1, 0.1, Array(""), 0)
  /*
  * Set the number of iteration to run. Default: 50.
  * */
  def setIteration(numIteration: Int)={
    this.maxIteration = numIteration
    this
  }

  /*
  * Set the number of topic to model. Default: 10.
  * */
  def setTopic(topicNum: Int)={
    this.numOfTopic = topicNum
    this
  }

  /*
  * Set alpha. Default: 0.1.
  * */
  def setAlpha(alphaValue: Double)={
    this.alpha = alphaValue
    this
  }
  /*
  * Set beta. Default: 0.1.
  * */
  def setBeta(betaValue: Double)={
    this.beta = betaValue
    this
  }

  /*
  * Set vocabulary, which is an array of String.
  * */

  def setVoc(voc: Array[String])={
    this.vocabulary = voc
    this.numOfVoc = voc.length
    this
  }

/*  def sample(array: Array[Double], samp: Double):Int= {
    var arraySum = array
    var topic = 0
    var find = true
    if(samp>array(0)){
      while(find&&topic<array.length){
        topic+=1
        arraySum(topic)+=arraySum(topic-1)
        if(samp<arraySum(topic)){
          find = false
        }
      }
    }
    topic
  }*/

  def sample(array:Array[Double], s:Double):Int={
    var sum = 0.0
    var topic = 0
    var find = true
    while(find&&topic<array.length){
      sum+=array(topic)
      if(s<sum){
        find=false
      }else{
        topic+=1
      }
    }
    topic
  }

  /*
  * Sets whether to use breeze for calculating global count.
  * Default: true.
  * */
  var useBreeze = true
  def setBreeze(a: Boolean)={
    this.useBreeze = a
    this
  }

  /*
  * Update global count.
  * */
  def updateGlobalCount(iterator: RDD[LDAModel]): Array[Array[Int]]={
    if(useBreeze){
      /*iterator.map(model=>{
        var count = new Array[SparseVector[Int]](numOfTopic)
        for(i<-0 until count.length){
          count(i) = new SparseVector[Int](new SparseArray[Int](numOfVoc))
        }
        for(i<- 0 until model.topic.length){
          count(model.topic(i))(model.word(i))+=1
        }
        count
      }).reduce((x,y)=>{
        var total = new Array[SparseVector[Int]](numOfTopic)
        for(i<-0 until total.length){
          total(i) = new SparseVector[Int](new SparseArray[Int](numOfVoc))
        }
        for(i<- 0 until total.length){
          total(i)= x(i)+y(i)
        }
        total
      }).map(x=>x.toArray)*/
      iterator.glom().map(array=>{
        var count = new Array[SparseVector[Int]](numOfTopic).map(x=> SparseVector.zeros[Int](numOfVoc))
        array.foreach(model=>{
          for(i<- 0 until model.topic.length){
            count(model.topic(i))(model.word(i))+=1
          }
        })
        count
      }).reduce((x,y)=>{
        x.zip(y).map(xx=> xx._1 + xx._2)
      }).map(x=>x.toArray)
    }else{
      iterator.map(model => {
        var count = Array.ofDim[Int](numOfTopic, numOfVoc)
        for (i <- 0 until model.word.length) {
          count(model.topic(i))(model.word(i)) += 1
        }
        count
      }
      ).reduce((x, y) => {
        var newArray = Array.ofDim[Int](numOfTopic, numOfVoc)
        for (i <- 0 until x.length) {
          for (j <- 0 until x(i).length) {
            newArray(i)(j) = x(i)(j) + y(i)(j)
          }
        }
        newArray
      })
    }
  }

  /*
  * Run distributed inference for LDA.
  * Return a Result class with all count information
  * */

  def run(initialData: RDD[Data]): Result= {
    require(vocabulary.length>0)
    val sc = initialData.sparkContext
    var iterator = initialData.map(data =>
      LDAModel(data.ID, data.word, new Array[Int](data.word.length), {
        var gamma = Array.ofDim[Double](data.word.length, numOfTopic)
        val initial = 1.0 / numOfTopic.toDouble
        gamma.map(x => x.map(xx => initial))
      }, new Array[Int](numOfTopic)))

    var topicTermCount = Array.ofDim[Int](numOfTopic, numOfVoc)
    var topicCount = new Array[Int](numOfTopic)

    var numIteration = 0

    while (numIteration < maxIteration) {
      numIteration += 1


      val nextIter = iterator.mapPartitions(x => {
        var rand = new Random()
        x.map { model => LDAModel(model.ID, model.word, {
          var nextTopic = new Array[Int](model.topic.length)
          for (i <- 0 until model.topic.length) {
            nextTopic(i) = sample(model.gamma(i), rand.nextDouble())
          }
          nextTopic
        }, model.gamma,{
          var nextCount = new Array[Int](numOfTopic)
          for (k <- model.topic) {
            nextCount(k) += 1
          }
          nextCount
        })
        }
      }).persist

      topicTermCount = updateGlobalCount(nextIter)
      val ttCount = sc.broadcast(topicTermCount)
      topicCount = topicTermCount.map(x => x.sum)
      val tCount = sc.broadcast(topicCount)
      iterator.unpersist()
      iterator = nextIter.map (model =>
        LDAModel(model.ID, model.word, model.topic, {
          var nextGamma = Array.ofDim[Double](model.word.length, numOfTopic)
          for (i <- 0 until model.word.length) {
            for (k <- 0 until numOfTopic) {
              if (model.topic(i) == k) {
                nextGamma(i)(k) = (alpha + model.count(k) - 1) *
                  (beta + ttCount.value(k)(model.word(i)) - 1) /
                  (numOfVoc * beta + tCount.value(k) - 1)
              } else {
                nextGamma(i)(k) = (alpha + model.count(k)) *
                  (beta + ttCount.value(k)(model.word(i))) /
                  (numOfVoc * beta + tCount.value(k))
              }
            }
          }
          nextGamma.map(x => (x, x.sum)).
            map { case (x, y) => x.map(xx => xx / y)}
        }, model.count)
      ).setName("Iter-"+numIteration).
        persist(StorageLevel.MEMORY_AND_DISK)

      if((!sc.getCheckpointDir.isEmpty) && (numIteration%10==0)) {
        iterator.checkpoint()
      }

      iterator.count()

      nextIter.unpersist()
      ttCount.unpersist()
      tCount.unpersist()
    }

    val topicVocProb = topicTermCount.map(x=>{
      val sum = x.sum
      x.map(xx=>(xx+beta)/(numOfVoc*beta+sum))
    })

    val docTopicProb = iterator.map(model=>(model.ID,{
      val sum = model.count.sum
      model.count.
        map(x=>(x+alpha)/(numOfTopic*alpha+sum))
    }))
    docTopicProb.count()
    new Result(docTopicProb, topicVocProb, vocabulary)
  }
}

/*
* Class Result represents the result of LDA inference
* */

class Result(docTopicProb: RDD[(String, Array[Double])],
             topicVocProb: Array[Array[Double]],
             vocabulary: Array[String]) extends Serializable {

  /*
  * Get document-topic probability matrix
  * */
  def getDocTopicProb = docTopicProb


  /*
  * Get topic-word probability matrix,
  * */
  def getTopicVocProb = topicVocProb

  /*
  * Set the number of words that represents
  * */
  var n = 20
  def setN(n: Int)={
    this.n=n
    this
  }
  /*
  * Set the threshold value of probability. If docTopicProb(d)(k) is bigger than this value, then
  * document d is related to topic k.
  * */
  var prob = 0.2
  def setProb(prob: Double)={
    this.prob = prob
    this
  }

 /*
 * Get the result of Topic-word probability matrix Beta, Beta(k)(v) represents the probability of word v presents
 * under topic k.
 * */

  def getBeta(): Array[Array[Double]]={
    topicVocProb
  }

  /*
  * Get the result of document-topic probability matrix as a map, Alpha(d) represents the probability that
  * document d belongs to each topic.
  * */

  def getAlpha(): scala.collection.Map[String, Array[Double]]={
    docTopicProb.collectAsMap()
  }

  def maxNIndex(n: Int, array: Array[Double]): Array[(Int, Double)]={
    var sort = new Array[(Int,Double)](array.length)
    for(i<- 0 until array.length){
      sort(i)=(i, array(i))
    }
    sort.sortWith((x,y)=>(if(x._2>y._2) true else false))
  }

  /*
  * Get the largest n in probability words for each topic
  * */

  def getTopicFeature(): Array[Array[String]]={
    topicVocProb.map(x=>maxNIndex(n,x).map(_._1)).
      map(x=>x.map(xx=>vocabulary(xx)))
  }

  /*
  * Get feature topic for each document, if oneFeature equals to true, then only get one feature with
  * largest probability, else get all features that the probability is larger than prob.
  * */

  var oneFeature = true
  def setFeature(isOneFeature: Boolean)={
    this.oneFeature = isOneFeature
    this
  }
  def getDocFeature():RDD[(String,Array[Boolean])]={
    if(oneFeature){
      docTopicProb.map(x=>(x._1, {
        var feature = x._2.map(_=>false)
        feature(maxNIndex(1,x._2)(0)._1)=true
        feature
      }))
    }else{
      docTopicProb.map(x=>(x._1,x._2.map(_>prob)))
    }
  }

  /*
  * Save topic features and document features in a file
  * */
  def saveFeatures(path: String) = {
    val fw = new OutputStreamWriter(new FileOutputStream(path),"UTF-8")
    val topicFeature = getTopicFeature()
    val docFeature = getDocFeature()
    val numOfTopics = topicFeature.length
    for(i<- 0 until numOfTopics){
      fw.write("Topic"+(i+1)+"\t")
      topicFeature(i).foreach(x=>fw.write(x+"\t"))
      fw.write("\r\n")
    }
    fw.write("\r\n\t")
    (1 to numOfTopics).map(x=>fw.write("Topic"+x+"\t"))
    docFeature.collect.foreach(line=>{
      fw.write("\r\n")
      fw.write(line._1+"\t")
      line._2.foreach(if(_) fw.write("Y\t") else fw.write("\t"))
    })
    fw.close()
  }
}