// =======================================================================
// Institut Mines Telecom - Teralab
// =======================================================================
package fr.teralabdatascience.tests


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object BreastCancer {
  var path: String = _
  var num_iter: Int = 20
  var numIterations: Int = 2

  // Print argument list
  def usage() {
    println("Arguments :")
    println("    path     : HDFS path for CSV file from Winsconsin Breast Cancer test case")
    println("    num_iter : K-Means iteration number (default to 20")
  }

  // Print argument list
  def test_arguments(args: Array[String]) {
    if ( (args.length < 1) || ( args(0).length <= 0) ) {
      usage()
      System.exit(1)
    }
    path = args(0)
    if (args.length > 1) {
      numIterations = args(1).toInt
    }
  }

  // Function to skip beginning of Array
  def tail(a: Array[String], h: Int) = { a.slice(h, a.length) }

  // Main method
  def main(args: Array[String]) {
    // Arguments testing
    test_arguments(args);

    // Spark initialisation
    val sparkConf = new SparkConf().setAppName("BreastCancer")
    val sc = new SparkContext(sparkConf)

    // Convert CSV file to Vectors usable for MLlib
    val data = sc.textFile(path).map {
      line => Vectors.dense(tail(line.split(','),2).map(_.toDouble))
    }.cache()

    // Calculate
    val numClusters = 2
    val clusters = KMeans.train(data, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(data)
    println("Within Set Sum of Squared Errors = " + WSSSE)
  }

  // Construct a kind of confusion matrix
  val result = sc.textFile("./wisconsin_data_breast_cancer_data_org.csv").map { line =>
        val items = line.split(",")
 		val point = Vectors.dense(tail(items,2).map(_.toDouble))
        val prediction = clusters.predict(point)
       (prediction, items(1))
  }
  val keys = List((0,"M"), (1, "M"), (0,"B"), (1, "B"));
  val matrix = result.countByValue();
  print("  | M   B")
  for (r <- 0 to 1) {
	print("r | ")
    List("M", "B").foreach { c =>
	  print(matrix.get((r,c)) + "  ")
	}
	println("")
  }

}

// vim: sw=2:ts=2:ai

