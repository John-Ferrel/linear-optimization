The Dual Simplex Method
=================================================================
### Abstract

This is a code implementation based on dual simplex  method. It can only deal with linear programming problems in standard form.

It must start with all reduced costs nonnegative.

---

### Development Environment

```
Window 10 21H2
python 3.8.8
numpy 1.20.1
```


---------------
### Input Format

In this project, I provide 4 input options.
    0: Use default problem, and disable input
    1: Execute normal mode, input data in the window
    2: Input file name, use file data
    3: Use example file, and disable input

0: the default problem is re-written by Exercise 3.17( Page.132).
```    
c = [-3, 1, 0, 0, 0]
b = [13/7, 9/7, 31/7]
basis_index = [0, 1, 3]
A = 
[1, 0, 1/7, 0, 2/7;
0, 1, -2/7, 0, 3/7;
0, 0, -3/7, 0, 22/7]
```

1: Enter the problem in the window. There are input prompt.

2: Input file name.

​    such as 

    example.txt

​    In the file, data format should be:

```
-3, 1, 0, 0, 0
13/7, 9/7, 31/7
0, 1, 3
1, 0, 1/7, 0, 2/7
0, 1, -2/7, 0, 3/7
0, 0, -3/7, 0, 22/7
```

​    The fist line is  $c^\top$.
​    The second line is $b^\top$.
​    The third line is basis_index.
​    The rest lines are rows of $A_{m\times n}$.

3: There is already a example file. And the content of the document is as above.

---

### Output Format

​    example:

```
opt_x: [1.85714286 1.28571429 0.         4.42857143 0.        ]
opt_v: -4.285714285714286
```

---

###### The origin problem:

![image-20220514135301855](C:\Users\MSIK\AppData\Roaming\Typora\typora-user-images\image-20220514135301855.png)

















