from statsmodels.tsa.seasonal import STL
import pandas as pd

# This script provides a function to decompose a time series into its
# trend, seasonal, and residual components using STL (Seasonal-Trend decomposition using LOESS).
# model: "additive" or "multiplicative" (default is "additive").
# explain model parameter:
# - "additive": assumes that the components add together to form the time series.
# - "multiplicative": assumes that the components multiply together to form the time series.
def decompose_series(y: pd.Series, model="additive", period=None, robust=True):
  """
  Returns trend, seasonal, resid using STL if period is given,
  otherwise falls back to STL's automatic period detection.
  """
  # period is the number of observations per cycle (e.g., 12 for monthly data)
  # robust=True makes the decomposition robust to outliers
  # seasonal is set to 13 to approximate the LOESS window size
  stl = STL(y, period=period, robust=robust, seasonal=13)  # seasonal=13 ≈ LOESS window
  res = stl.fit()
  return res.trend, res.seasonal, res.resid

if __name__ == "__main__":
  # give an Time Series in pandas Series format
  y = pd.Series([1, 2, 3, 4, 5])
  print(f"input series: {y}")
  trend, seasonal, resid = decompose_series(y, model="additive", period=3)
  print(f"trend: {trend}")
  print(f"seasonal: {seasonal}")
  print(f"residual: {resid}")