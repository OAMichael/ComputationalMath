# Лабораторная работа №4
## Различные методы интерполяции функций


#### В работе интерполируется и экстраполируется функция, заданная таблично — зависимость населения США в период 1910-2000 гг. Экстраполяция проводится к 2010 году. Это задание **VI.9.32**. Используются следующие методы: многочлен Ньютона и сплайн-интерполяция, точнее — естественный сплайн. 

#### Также, с помощью сплайна были интерполированы к нужной точке таблично заданные функции из номера **VI.9.28**.

#### На основании номера **VI.9.32** проводилось сравнение методов интерполяции, используемых в нем.

#### Подробнее — в отчете.

#### Вывод скрипта в консоль выглядит так:
```console
==================================================================
| x ||  0.00000  |  1.00000  |  2.00000  |  3.00000  |  4.00000  |
==================================================================
| f ||  0.00000  |  0.50000  |  0.86603  |  1.00000  |  0.86603  |
==================================================================
Interpolation at x* = 1.5: f(1.5) = 0.7061482812500001


==================================================================
| x ||  0.00000  |  0.50000  |  0.90000  |  1.30000  |  1.70000  |
==================================================================
| f || -2.30260  | -0.69315  | -0.10536  |  0.26236  |  0.53063  |
==================================================================
Interpolation at x* = 0.8: f(0.8) = -0.21275445989173225


==================================================================
| x ||  0.00000  |  1.70000  |  3.40000  |  5.10000  |  6.80000  |
==================================================================
| f ||  0.00000  |  1.30380  |  1.84390  |  2.25830  |  2.60770  |
==================================================================
Interpolation at x* = 3.0: f(3.0) = 1.7531560510017157


==================================================================
| x || -0.40000  | -0.10000  |  0.20000  |  0.50000  |  0.80000  |
==================================================================
| f ||  1.98230  |  1.67100  |  1.36940  |  1.04720  |  0.64350  |
==================================================================
Interpolation at x* = 0.1: f(0.1) = 1.4694391534391535


==================================================================
| x ||  0.00000  |  1.00000  |  2.00000  |  3.00000  |  4.00000  |
==================================================================
| f ||  1.00000  |  1.54030  |  1.58390  |  2.01000  |  3.34640  |
==================================================================
Interpolation at x* = 1.5: f(1.5) = 1.5862379464285716


------------------- USA population prediction --------------------
Prediction by 2010 using Newton polynomial: 827906509
Prediction by 2010 using Spline interpolation: 314133939
```