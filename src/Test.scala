import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.{SparkContext, SparkConf}

class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[LDA])
    kryo.register(classOf[Result])
  }
}

object Test{
  def main(args: Array[String]) {
    var topicK =15
    var numOfPartitions = 60
    var alpha = 0.1
    var beta = 0.1
    var maxIteration = 50
    var input = ""
    var output = ""
    var vocPath=""
    var master=""

    if(args.length==4){
      input = args(0)
      output = args(1)
      vocPath = args(2)
      master = args(3)
    }

    val configure = new SparkConf().
      setMaster(master).
      setAppName("LDATest").
      set("spark.executor.memory", "1024m").
      set("spark.cores.max", "10")
    configure.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    configure.set("spark.kryo.registrator", "MyRegistrator")

    val sc = new SparkContext(configure)
    val vocabulary = sc.textFile(vocPath).collect()
    val data = sc.textFile(input,numOfPartitions).
      map(x=>x.split(":")).
      map(x=>(x(0),x(1).split(" "))).
      map{case(x,y)=>new Data(x,y.map(yy=>yy.toInt))}
    val lda = new LDA().
      setAlpha(alpha).
      setBeta(beta).
      setIteration(maxIteration).
      setVoc(vocabulary).
      setTopic(topicK)
    val result = lda.run(data)
    result.saveFeatures(output)
  }
}