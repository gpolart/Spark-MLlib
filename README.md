# Spark and MLlib test programs

---
## GOALS ##

This package provides some Scala examples to use MLLib to process some well known datasets.
It's part of a work realized for Institut Mines-Telecom in Teralab project : https://teralab-datascience.fr.

## Breast Cancer Wisconsin ##

I use this dataset for KMeans clustering with verification from the diagnostic. It's a small dataset (569 lines) usefull for testing purpose. The CSV file is used directly and vectorized by Scala program.A So we need only to put CSV file on HDFS and call the spar-summit programm with arguments :

`spark-submit --class fr.teralabdatascience.tests.BreastCancer --deploy-mode client --master yarn target/tests-0.0.1-SNAPSHOT.jar wisconsin_data_breast_cancer_data_org.csv`

- Diagnostic dataset : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
- Original dataset : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

## Higg's boson dataset ##

Higg's boson is a big dataset (8Go) given as a CSV file. We use the same program as for Breast Cancer Wisconsin with a litlle difference : there is only one first col for classification result (0 or 1).

Dataset : http://archive.ics.uci.edu/ml/datasets/HIGGS



