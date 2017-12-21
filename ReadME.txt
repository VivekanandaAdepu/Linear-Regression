1. Make sure you have the Coudera installed in your VM.

2. Cloudera provides a default python version as 2.6.6 and also Spark.

3. As we are trying to implement linear regression using pyspark, we use the package named 'numpy'.

4. Inorder to install numpy on the python we have to install pip first. 

5. But, pip  is not supported by the default python 2.6.6 version provided by cloudera.

6. Hence, we install python version 3 to cloudera and then install pip.

7. We change the default version of python used by spark from 2.6.6 to 3 by using the following command:
 
	export PYSPARK_PYTHON=python3

8. Now we install numpy package on python3.

9. Run the linreg.py program using the following command:

	spark-submit <filename> <input> <output>

		Example : spark-submit linreg.py /user/cloudera/linear/yxlin.csv yxlin.out

10. The compiler takes the input as csv file and writes the output to the yxlin.out file.

11. The .out file contains the beta coefficients.

12. The equation used to compute beta is:

	Beta=(Inverse(X_Transpose.X)).(X_Transpose.Y)

		'.' in the above equation represents 'dot product'

