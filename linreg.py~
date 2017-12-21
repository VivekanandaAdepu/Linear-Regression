# Vivekananda Adepu
# 800967951
#
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x and the output is printed in a .out file.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv

import sys
import numpy as np
from numpy.linalg import inv 								# To implement the inverse function
from pyspark import SparkContext

def getXXt(lines):									# A function to find dot product of X_Transpose and X 
	line = lines.split(',')								# Split the contents of csv file using ','
	line[0] = 1
	x = np.array([line], dtype=float)
	xt = x.transpose()								# Transpose function is used to find the transpose of x
	return xt*x

def getXtY(lines):									# A Function to find the dot product of X_transpose and Y
	line = lines.split(',')
	y=float(line[0])
	line[0] = 1
	xt = np.array([line], dtype=float).transpose()
	return xt*y 

if __name__ == "__main__":
  if len(sys.argv) !=3:
    print (sys.stderr, "Usage: Improper Arguments: Expected minimum::3 ")
    exit(-1)

  sc = SparkContext(appName="LinReg")

  yxinputFile = sc.textFile(sys.argv[1])						# The input file is converted into a textfile before it is used

  xxt = inv(yxinputFile.map(getXXt).reduce(lambda a,b: np.add(a,b)))			# we use reduce function to find the summation and inv to find the inverse

  yxt = yxinputFile.map(getXtY).reduce(lambda a,b: np.add(a,b))

  beta = np.dot(xxt,yxt)								# The dot product of the yxt and xxt is found using dot function in numpy package

  temp = sys.stdout									# The original functionality of sysstdout is preserved in a temporary variable
  output = open(sys.argv[2], 'w')							# The output file title is read from the command line and a new file is created 
  sys.stdout = output									# The sys.stdout is given a new definition to print in output file
  
  print( "beta:")
  for coeff in beta:
      print( coeff)									# The dot product gives the linear coefficients which are stored in beta is printed into the output file
  
  sys.stdout = temp									# The original functionality of sys.stdout is restored 
  output.close()
  sc.stop()
