The Two-phase Simplex Method

### Abstract

This is a code implementation based on simplex two-phases method. It can only deal with linear programming problems in standard form.

In the first phase, auxiliary variables are added to determine whether the problem has a solution. If there is a solution, it will find a basic feasible solution of the original problem.

In the second phase, the optimal solution is found by tableau method.

---

### Development Environment

```
Window 10 21H2
python 3.8.8
numpy 1.20.1
```

---

### Input Format

In this project, I provide 4 input options.
 0: Use default problem, and disable input
 1: Execute normal mode, input data in the window
 2: Input file name, use file data
 3: Use example file, and disable input

0: the default problem is re-written by Exercise 3.17( Page.132).

```
c = [2, 3, 3, 1, -2]
A =[[1, 3, 0, 4, 1], [1, 2, 0, -3, 1], [1, 4, -3, 0, 0]]
b = [2, 2, -1]
```

1: Enter the problem in the window. There are input prompt.

2: Input file name.

​ such as

```
example.txt
```

​ In the file, data format should be:

```
2, 3, 3, 1, -2
2, 2, -1
1, 3, 0, 4, 1
1, 2, 0, -3, 1
1, 4, -3, 0, 0
```

​ The fist line is $c^\top$.
​ The second line is $b^\top$
​ The rest lines are rows of$ A_{m\times n}$

3: There is already a example file. And the content of the document is as above.

---

### Output Format

opt_x: [2.35649152e-11 5.22871972e-17 3.33333333e-01 1.74084265e-15 2.00000000e+00]
opt_v: -2.999999999882185

---


