# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the IOT-temp.xls dataset
data = pd.read_excel('/path/to/IOT-temp.xls')  
print(data.head()) 


X = data.iloc[:, 1]

# Set figure size
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

# Plot the original data
plt.plot(X)
plt.title('Original IoT Temperature Data')
plt.show()

# Plot ACF and PACF for the original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# Fit ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# Fit ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()


```

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/user-attachments/assets/92635e79-fa35-45fd-8686-7e52e7d0b876)


Partial Autocorrelation
![image](https://github.com/user-attachments/assets/1ccbea98-81e8-4394-aab8-aa8140999a03)

Autocorrelation
![image](https://github.com/user-attachments/assets/814683de-bb8a-4857-8acc-11543f279e7f)



SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/6f6134eb-c43d-401a-bc63-4c879381ef16)

Partial Autocorrelation
![image](https://github.com/user-attachments/assets/0cd55462-af3c-4fe2-8a19-ad61c1cdad29)



Autocorrelation
![image](https://github.com/user-attachments/assets/2fca5398-a9a4-40f3-b3bf-fbac5f75b867)

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
