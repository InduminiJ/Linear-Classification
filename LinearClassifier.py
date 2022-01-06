### David Neudorf, Indumini Jayakody, Helen Lin
### 2021-10-10
### Comp4105 Mini-project 1
import pandas as pd, numpy as np
from scipy.special import expit

class LogisticRegression:
  def __init__(self, learning_rate=0.0001, epsilon=0.000001): # put these hyperparameters into the fit function
    self.learning_rate = learning_rate
    #self.iterations = iterations
    self.epsilon = epsilon
    self.weights = None
    self.bias = None

  #takes data set X and true value set Y, generates weights using classes learning rate and epsilon
  def fit(self, X, y):
    # init parameters
    samples, features = X.shape
    #self.weights =  np.random.random_sample(size = features) #accuracy got worse with rnd start weights, which is odd
    self.weights = np.zeros(features)
    self.bias = 0

    c = 0
    # gradient descent
    lr = self.learning_rate
    alpha_dec = 1000
    while True:
      c+=1
      oldWeights = self.weights.copy()
      linear_model = np.dot(X, self.weights) + self.bias
      y_predicted = self._sigmoid(linear_model)

      dw = (1/samples) * np.dot(X.T, (y_predicted - y))
      db = (1/samples) * np.sum(y_predicted - y)

      self.weights -= lr * dw
      self.bias -= lr * db
      delta = np.subtract(self.weights, oldWeights)
      if(delta.dot(delta) < self.epsilon): #or c>10000):
        #print("iterations: ", c)
        #print("learning rate: ", lr)
        break
      if(c>alpha_dec):
        alpha_dec+=750
        lr = max(lr*0.9,2*self.epsilon) #slowly decrement learning rate as we iterate, always greater than epsilon

  #given data X, predict the output class
  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_predicted = self._sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_cls

  def _sigmoid(self, x):
    return expit(x) #needed this because data was too large, it does the exact same thing as commented function
    #return 1 / (1 + np.exp(-x)) fails for stock, overflow

#given a dataframe and array of ratios (sum to one), return a dataframe split via randomized mask selection
def partition(data, sizes, seed=0):
  #theoretically the randomness serves to shuffle, so we don't need to do that
  out=[]
  d = data
  np.random.seed(seed)
  amtLeft = 1
  for i in sizes:
    mask = np.random.rand(len(d))<(i/amtLeft)
    #print(len(d[mask]))
    out.append(d[mask])
    d = d[~mask]
    amtLeft -=i
  out.append(d)
  #print(len(d))
  return out

#returns the accuracy of the predictions
def accu_eval(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

#given a dataframe and number of folds, splits the data into k partitions and perfoms k-fold over them returning the avg_error of the model
def k_fold(df, k=10, learning_rate=0.0001,epsilon=0.00000001, seed=0):
  partitions = partition(df,[1/k]*(k-1),seed=seed)
  err = 0
  for i in range(k):
    #print("i is: ", i )
    temp_df = pd.concat(partitions[:i]+partitions[i+1:])
    #print("train df size: ", temp_df.shape)
    #print("test df size:  ", partitions[i].shape)
    X_test = partitions[i][df.columns[1:]]
    X_train = temp_df[df.columns[1:]]
    y_test = partitions[i]["Class Label"]
    y_train = temp_df["Class Label"]

    model = LogisticRegression(learning_rate=learning_rate, epsilon=epsilon)
    model.fit(X_train,y_train)
    #print("accuracy: ", accu_eval(y_test,model.predict(X_test)))
    err+=accu_eval(y_test,model.predict(X_test))
  return (k-err)/k

if __name__ == '__main__':
  df = pd.read_csv('Parkinson.csv')

  #split up all the discrete 0-3 data into binary columns
  preproc_df = pd.get_dummies(df, columns = ['18. Speech', '19. Facial Expression',
       '20. Tremor at Rest - head', '20. Tremor at Rest - RUE',
       '20. Tremor at Rest - LUE', '20. Tremor at Rest - RLE',
       '20. Tremor at Rest - LLE', '21. Action or Postural Tremor - RUE',
       '21. Action or Postural Tremor - LUE', '22. Rigidity - neck',
       '22. Rigidity - RUE', '22. Rigidity - LUE', '22. Rigidity - RLE',
       '22. Rigidity - LLE', '23.Finger Taps - RUE', '23.Finger Taps - LUE',
       '24. Hand Movements  - RUE', '24. Hand Movements  - LUE',
       '25. Rapid Alternating Movements - RUE',
       '25. Rapid Alternating Movements -  LUE', '26. Leg Agility - RLE',
       '26. Leg Agility - LLE', '27.  Arising from Chair ', '28. Posture',
       '29. Gait', '30. Postural Stability',
       '31. Body Bradykinesia and Hypokinesia'])
  #normalize the data by column
  normalized_pp_df=(preproc_df-preproc_df.min())/(preproc_df.max()-preproc_df.min())

  df_stock = pd.read_csv('Stock.csv')

  #replace NaNs (zeroes) with the column mean
  df_stock[df_stock.columns[1:]] = df_stock[df_stock.columns[1:]].replace(0,df_stock.mean(axis=0))
  #normalize the data
  normalized_stock_df=(df_stock-df_stock.min())/(df_stock.max()-df_stock.min())

  normalized_nopp_df = (df-df.min())/(df.max()-df.min())

  #current best
  parkinson_df_norm_badcolsdropped = normalized_nopp_df.drop(columns=['19. Facial Expression',
       '20. Tremor at Rest - head', '20. Tremor at Rest - RUE',
       '20. Tremor at Rest - LLE', '21. Action or Postural Tremor - LUE', '22. Rigidity - neck',
       '22. Rigidity - LUE', '22. Rigidity - RLE',
       '23.Finger Taps - RUE', '23.Finger Taps - LUE',
       '24. Hand Movements  - RUE', '24. Hand Movements  - LUE',
       '25. Rapid Alternating Movements - RUE',
       '25. Rapid Alternating Movements -  LUE', '26. Leg Agility - RLE',
       '26. Leg Agility - LLE','29. Gait',
       '31. Body Bradykinesia and Hypokinesia',
       'Decay of unvoiced fricatives              (promile/min)',
       'Latency of\nrespiratory exchange (ms)',
       'Entropy of speech timing (-).1',
       'Decay of unvoiced fricatives              (?/min)',
       'Relative loudness of respiration (dB).1',
       'Pause intervals per respiration (-).1',
       'Latency of\nrespiratory exchange (ms).1'])

  #print(k_fold(normalized_stock_df.drop(columns= ['MEBL_low', 'OGDC_low', 'BERG_average', 'BERG_high', 'BERG_close', 'EPCL_low', 'AMBL_high', 'GATM_ldcp', 'FFLNV_close', 'FFLNV_open', 'FFLNV_ldcp']), k=10, learning_rate=0.1, epsilon=0.000001, seed=0))
  #print(k_fold(normalized_stock_df[normalized_stock_df.columns[~normalized_stock_df.columns.str.endswith('_close')]], k=10, learning_rate=0.1, epsilon=0.000001, seed=0))

  #df = df[df.columns[~df.columns.str.endswith('_o')]]]

  helen_df = normalized_nopp_df.drop(columns=['Age (years)', 'Gender',
       'Medication - Levodopa equivalent (mg/day)',
       '18. Speech',
       '20. Tremor at Rest - head', '20. Tremor at Rest - RUE',
       '20. Tremor at Rest - RLE',
       '20. Tremor at Rest - LLE', '21. Action or Postural Tremor - RUE',
       '21. Action or Postural Tremor - LUE', '22. Rigidity - neck',
       '22. Rigidity - LUE', '22. Rigidity - RLE',
       '22. Rigidity - LLE', '23.Finger Taps - LUE',
       '24. Hand Movements  - RUE', '24. Hand Movements  - LUE',
       '25. Rapid Alternating Movements - RUE',
       '25. Rapid Alternating Movements -  LUE', '26. Leg Agility - RLE',
       '26. Leg Agility - LLE', '27.  Arising from Chair ', '28. Posture',
       '29. Gait', '30. Postural Stability',
       '31. Body Bradykinesia and Hypokinesia',
       'Acceleration of speech timing                                (-/min2)',
       'Duration of unvoiced stops (ms)',
       'Decay of unvoiced fricatives              (promile/min)',
       'Pause intervals per respiration (-)',
       'Rate of speech respiration                (-/min)',
       'Latency of\nrespiratory exchange (ms)',
       'Entropy of speech timing (-).1',
       'Duration of pause intervals (ms).1',
       'Duration of voiced intervals (ms).1',
       'Gaping                         in-between voiced\nintervals                   (-/min)',
       'Duration of unvoiced stops (ms).1',
       'Relative loudness of respiration (dB).1',
       'Pause intervals per respiration (-).1',
       'Rate of speech respiration           (- /min)',
       ])
  helen_df = normalized_nopp_df.drop(columns=[
       '20. Tremor at Rest - head', '20. Tremor at Rest - RUE',
       '20. Tremor at Rest - RLE',
       '20. Tremor at Rest - LLE', '21. Action or Postural Tremor - RUE',
       '21. Action or Postural Tremor - LUE', '22. Rigidity - neck',
       '22. Rigidity - LUE', '22. Rigidity - RLE',
       '22. Rigidity - LLE', '23.Finger Taps - LUE',
       '24. Hand Movements  - RUE', '24. Hand Movements  - LUE',
       '25. Rapid Alternating Movements - RUE',
       '25. Rapid Alternating Movements -  LUE', '26. Leg Agility - RLE',
       '26. Leg Agility - LLE'
       ])
  #print(k_fold(helen_df, k=10, learning_rate=0.1, epsilon=0.000001, seed=0))
  outdf = normalized_stock_df
  sigcols = ['ASTL_turnover', 'MEBL_turnover', 'SMCPL_turnover', 'OGDC_turnover', 'SERF_turnover', 'QUET_turnover', 'BERG_turnover', 'EPCL_turnover', 'AMBL_turnover', 'GATM_turnover', 'UBL_turnover', 'CPPL_turnover', 'JPGL_turnover', 'JSML_turnover', 'SRSM_turnover', 'JSCL_turnover', 'NPL_turnover', 'FFLNV_turnover', 'YOUW_turnover']
  for c in normalized_stock_df.columns:
    '''
    if(c.endswith("_close")):
      s = c[:-6]
      k = s+"_delta"
      kwargs = {k: normalized_stock_df[s+"_close"]-normalized_stock_df[s+"_open"]}
      outdf = outdf.assign(**kwargs)
    '''
    if(c.endswith("_average")):
      s = c[:-8]
      k = s+"_pow0.1"
      kwargs = {k: normalized_stock_df[s+"_average"]**0.1}
      outdf = outdf.assign(**kwargs)
    if(c in sigcols):
        k = c+"_pow1.5"
        kwargs = {k: normalized_stock_df[c]**1.5}
        outdf = outdf.assign(**kwargs)
  print(k_fold(outdf, k=10, learning_rate=0.1, epsilon=0.000001, seed=0))
'''
Baseline stock normalized with zeroes replaced by column mean/median (same result)
0.188

breakpoints
<=0.188 --> drop
>=0.195 --> do shit


DROP
0.18775277337859783
MEBL_low

0.18827318306889912
OGDC_low

0.18844676078780562
BERG_average

0.1880709755671143
BERG_high

0.1883394134972381
BERG_close

0.18847276643483113
EPCL_low

0.18792957012355488
AMBL_high

0.18806584168207455
GATM_ldcp

0.1883534236237617
FFLNV_close

0.1883780009292762
FFLNV_open

0.18746307106715054
FFLNV_ldcp




DO SHIT
0.20900795935118052
ASTL_turnover

0.20096198914005498
MEBL_turnover

0.20302451531667964
SMCPL_turnover

0.2074087974053195
OGDC_turnover

0.20484002811546825
SERF_turnover

0.20052537642463406
QUET_turnover

0.20404661240984154
BERG_turnover

0.2028093234620215
EPCL_turnover

0.20133546995272766
AMBL_turnover

0.20081697685504815
GATM_turnover

0.20197794056795254
UBL_turnover

0.19934883900924927
CPPL_turnover

0.19483261886345585
JPGL_turnover

0.1979814026676104
JSML_turnover

0.20414609863329555
SRSM_turnover

0.19777139076587424
JSCL_turnover

0.2027348355718736
NPL_turnover

0.19752171389275386
FFLNV_turnover

0.20633646212202006
YOUW_turnover

----



'''
