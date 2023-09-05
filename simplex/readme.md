The Two-phase Simplex Method
=================================================================
### Abstract

This is a code implementation based on simplex two-phases method. It can only deal with linear programming problems in standard form.

In the first phase, auxiliary variables are added to determine whether the problem has a solution. If there is a solution, it will find a basic feasible solution of the original problem.

In the second phase, the optimal solution is found by tableau method.

---

### Environment

```
Window 10 21H2
python 3.8.8
numpy 1.20.1
```


---------------
### Input Format

In this project, I provide 4 input options.
    0: Use default problem, and disable input
    1: Execute normal mode
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

​    such as 

    example.txt

​    In the file, data format should be:

```
2, 3, 3, 1, -2
2, 2, -1
1, 3, 0, 4, 1
1, 2, 0, -3, 1
1, 4, -3, 0, 0
```

​    The fist line is  $c^\top$.
​    The second line is $b^\top$
​    The rest lines are rows of $A_{m\times n}$

3: There is already a example file. And the content of the document is as above.

---

### Output Format

​    example:

```
opt_x: [0.         0.         0.33333333 0.         2.        ]
opt_v: -3.0
```

---

### Implementation details

​	1.Choosing enters with random
​    Greedy policy can lead to cycles. To solve this problem, select randomly from the columns with reduced cost < 0 each time.

​    2.Forcing out the auxiliary
​    When the auxiliary variable = 0 but the corresponding column is still in the basis, all reduced costs have been > 0.
​    At this time, It is necessary to force the corresponding column of auxiliary variable as exits.

















