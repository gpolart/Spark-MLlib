// =======================================================================
// Institut Mines Telecom - Teralab
// =======================================================================
package fr.teralabdatascience.tests


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object Boson {
  var path: String = _
  var numIterations: Int = 2

  // Print argument list
  def usage() {
    println("Arguments :")
    println("    path       : HDFS path for CSV file from Higgs boson test case")
    println("    iterations : K-Means iteration number (default to 20")
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
    val sparkConf = new SparkConf().setAppName("Boson")
    val sc = new SparkContext(sparkConf)

    // Convert CSV file to Vectors usable for MLlib
    val data = sc.textFile(path).map {
      line => Vectors.dense(tail(line.split(','), 1).map(_.toDouble))
    }.cache()

    // Calculate
    val numClusters = 2
    val clusters = KMeans.train(data, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(data)
    println("Within Set Sum of Squared Errors = " + WSSSE)

		// Construct a kind of caracterisation matrix
		val result = sc.textFile(path).map { line =>
					val items = line.split(",")
			    val point = Vectors.dense(tail(items,1).map(_.toDouble))
					val prediction = clusters.predict(point)
				 (prediction, items(0).toDouble.toInt)
		}
		val keys = List((0,1), (1, 1), (0,0), (1, 0));
		val matrix = result.countByValue();
		println(matrix)
		println("  | 1   0")
		for (r <- 0 to 1) {
			print(r + " | ")
			List(1, 0).foreach { c => print(matrix(r,c) + "  ") }
			println("")
		}
  }
}

// vim: sw=2:ts=2:ai

