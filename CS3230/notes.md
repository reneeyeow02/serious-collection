# CS3230

## Correctness

- **Loop Invariant**: true at beginning of an iteration, and remains true at the beginning of the next iteration

### Iterative Algorithms

1. **Initialisation**: invariant is true before the first iteration of the loop
2. **Maintenance**: invariant remains true before the next iteration
3. **Termination**: when the algo terminates, the invariant provides a useful property for showing correctness

### Recursive Algorithms

Prove by **strong induction**, eg. with a statement $P(n)$,

1. **Base case**: prove base case $P(k_0)$ is true
    - $k_0$ is commonly $0$ or $1$
2. **Induction step**: prove $P(k+1)$ is true under the assumption that $P(k_0),P(k_0+1),\dots,P(k)$ are all true

## Efficiency

### Simplicity vs. Efficiency

- When the problem only occurs a few times and is small, we prefer a simple algorithm
- When the problem occurs many times and is big, we prefer an efficient algorithm

### Analysis

- Big $O$: asymptotic upper bound (may or may not be tight)
  - $f(n) = O(g(n))$ if $\exists$ $c, n_0 > 0$ st. $0 \leq f(n) \leq cg(n)$, $\forall n\geq n_0$
- Big $\Omega$: asymptotic lower bound
  - $f(n) = \Omega(g(n))$ if $\exists$ $c, n_0 > 0$ st. $0 \leq cg(n) \leq f(n)$, $\forall n\geq n_0$
- Big $\Theta$: asymptotic tight bounds (may or may not be tight)
  - $f(n) = \Omega(g(n))$ if $\exists$ $c_1, c_2, n_0 > 0$ st. $0 \leq c_1g(n) \leq f(n) \leq c_2g(n)$, $\forall n\geq n_0$
- Small $o$: asymptotic upper bound (but not tight; strictly $<$)
  - $f(n) = o(g(n))$ if $\exists$ $c, n_0 > 0$ st. $0 \leq f(n) < cg(n)$, $\forall n\geq n_0$
- Small $\omega$: asymptotic lower bound (but not tight; strictly $>$)
  - $f(n) = \Omega(g(n))$ if $\exists$ $c, n_0 > 0$ st. $0 \leq cg(n) < f(n)$, $\forall n\geq n_0$

$\Theta(g(n)) = O(g(n)) \cap \Omega(g(n))$

## Recurrences & Master Theorem

### Logarithm Properties

- $a = b^{\log_b a}$
- $\log_c(ab) = \log_ca + log_cb$
- $log_ba^n = nlog_ba$
- $log_ba = \frac{\log_ca}{\log_cb}$
- $log_b\frac{1}{a} = -\log_ba$
- $log_ba = \frac{1}{log_ab}$
- $a^{\log_bc} = c^{\log_ba}$
- $\lg n = \Theta(\ln n) = \Theta(\log_{10}n)$
  - Base of logarithm does not matter in asymptotics
- $\lg(n!) = \Theta(n\lg n)$

### Exponentials

- $a^{-1}=\frac{1}{a}$
- $(a^m)^n = a^{mn}$
- $a^ma^n = a^{m+n}$
- $e^x \geq 1+x$
- $n^k=o(a^n), \forall k>0, a>1$
  - Any exponential function $a^n$ with base $a > 1$ grows faster than any polynomial function $n^k$

### Series

- Arithmetic: $\sum_{k=1}^{n}k = \frac{1}{2}n(n+1) = \Theta(n^2)$
- Geometric: $\sum_{k=1}^{\infty}x^k = \frac{1}{1-x}$, when $|x| < 1$
- Harmonic: $H_n = \sum_{k=1}^n\frac{1}{k} = \log n + O(1) = \Theta(\log n)$
  - $H_{\log(n)} = \sum_{k=1}^{\log(n)}\frac{1}{k} = \log(\ln n) + O(1)= \Theta(\log\log n)$
- Telescoping: $\sum_{k=0}^{n-1}(a_k-a_{k+1}) = a_0 - a_n$

### Factorials

- $n! = \sqrt{2\pi n}(\frac{n}{e})^n(1+\Theta(\frac{1}{n}))$
- $\omega(2^n) = n! = o(n^n)$
  - $n!$ is lower bounded by $2^n$ and upper bounded by $n^n$

### Limit

Assume $f(n), g(n) > 0$,

- $\lim_{n\rightarrow \infty}\frac{f(n)}{g(n)}=0 \implies f(n)=o(g(n))$
- $\lim_{n\rightarrow \infty}\frac{f(n)}{g(n)}<\infty \implies f(n)=O(g(n))$
- $0<\lim_{n\rightarrow \infty}\frac{f(n)}{g(n)}<\infty \implies f(n)=\Theta(g(n))$
  - As $n \rightarrow\infty$, $\frac{f(n)}{g(n)}$ converges to a defined number
- $\lim_{n\rightarrow \infty}\frac{f(n)}{g(n)}>0 \implies f(n)=\Omega(g(n))$
- $\lim_{n\rightarrow \infty}\frac{f(n)}{g(n)}=\infty \implies f(n)=\omega(g(n))$
- L'Hopital's: $\lim_{x\rightarrow \infty}\frac{f(x)}{g(x)} = \lim_{x\rightarrow \infty}\frac{f'(x)}{g'(x)}$

### Calculus

Assumes $\lg = \log_2$ unless otherwise stated,

- $\frac{d}{dx}\log_bx = \frac{1}{x\ln b}$
- $\frac{d}{dx}\lg \lg n = \frac{1}{n\ln 2 \ln n}$
- $\frac{d}{dx}\lg^px = \frac{p\ln^{p-1}x}{x\ln^p(p-1)}$
- $\frac{d}{dx}e^{nx} = ne^{nx}$

### Properties of Big $O$

#### Transitivity

- $f(n) = \Theta(g(n)) \wedge g(n) = \Theta(h(n)) \implies f(n) = \Theta(h(n))$
- $f(n) = O(g(n)) \wedge g(n) = O(h(n)) \implies f(n) = O(h(n))$
- $f(n) = \Omega(g(n)) \wedge g(n) = \Omega(h(n)) \implies f(n) = \Omega(h(n))$
- $f(n) = o(g(n)) \wedge g(n) = o(h(n)) \implies f(n) = o(h(n))$
- $f(n) = \omega(g(n)) \wedge g(n) = \omega(h(n)) \implies f(n) = \omega(h(n))$

#### Reflexivity

- $f(n) = \Theta(f(n))$
- $f(n) = O(f(n))$
- $f(n) = \Omega(f(n))$

#### Symmetry

- $f(n) = \Theta(g(n)) \iff g(n) = \Theta(f(n))$

#### Complementarity

- $f(n) = O(g(n)) \iff g(n) = \Omega(f(n))$
- $f(n) = o(g(n)) \iff g(n) = \omega(f(n))$

#### Useful Bounds

- $O(\lg(n!)) \equiv O(n\lg n) \ll O(n^2) \ll O((\lg n)!) \ll O(2^n) \ll O(n!)$
- $O(\lg n) \ll O(n^\epsilon)$ (for any $\epsilon > 0$, ie. $\epsilon = 0.1$)

### Master Theorem

$T(n) = aT(\frac{n}{b}) + f(n)$, where $a \geq 1$, $b > 1$ are constans and $f(n)$ is an asymptotically positive function.

1. $f(n) = O(n^{\log_ba-\epsilon})$ for some constant $\epsilon > 0 \implies T(n) = \Theta(n^{\log_ba})$
    - $f(n)$ must be **polynomially smaller** than $n^{\log_ba}$ by a factor of $n^\epsilon$ for some constant $\epsilon > 0$
2. $f(n) = \Theta(n^{\log_ba}\lg^kn)$ for some $k\geq 0 \implies T(n) = \Theta(n^{\log_ba}\lg^{k+1} n)$
3. $[f(n) = \Omega(n^{\log_ba+\epsilon})$ for some constant $\epsilon > 0] \wedge [af(\frac{n}{b}) \leq cf(n)$ for some constant $c < 1$ and all sufficiently large $n] \implies T(n) = \Theta(f(n))$
    - $f(n)$ must be **polynomially larger** AND satisfy condition that $af(\frac{n}{b}) \leq cf(n)$

Intuitively, we compare the function $f(n)$ with the function $n^{\log_ba}$ and the larger of the functions determine the solution. Eg. if $f(n)$ is smaller, then $f(n)$ is upper bounded by $n^{\log_ba}$ (which is larger) and thus, case 1 is the answer: $T(n) = \Theta(n^{\log_ba})$.

Master thorem **can fail** to work: these 3 cases do not cover all possibilities of $f(n)$. There is a gap between cases 2 and 3 when $f(n)$ is asymptotically larger, but not polynomially larger, than $n^{\log_ba}$.

#### Negative Example

Eg. for $T(n) = 2T(\frac{n}{2}) + n\lg n$, $f(n) = n\lg n$ is asymptotically larger than $n^{\log_ba} = n$ but **not polynomially larger** (and thus, **case 3 cannot apply**). Prove:

1. Let $g(n) = n^{\log_ba+\epsilon} = n^{1+\epsilon} = n \times n^\epsilon$, for some constant $\epsilon > 0$
2. Assume $f(n)$ is indeed polynomially larger than $g(n)$
    - ie. $\exists\,\epsilon>0$ st. $f(n) = \Omega(n^{\log_ba+\epsilon}) = \Omega(g(n))$
3. To find the asymptotic bounds, take $\lim_{n\rightarrow\infty}\frac{f(n)}{g(n)}$
    - $\frac{f(n)}{g(n)} = \frac{n\lg n}{n\times n^\epsilon} = \frac{\lg n}{n^\epsilon}$
    - $f'(n) = \frac{1}{n\ln 2}$, $g'(n) = \epsilon n^{\epsilon-1}$
    - Using L'Hopital's, $\lim_{n\rightarrow\infty}\frac{f(n)}{g(n)}=\lim_{n\rightarrow\infty}\frac{f'(n)}{g'(n)} = \lim_{n\rightarrow\infty}\frac{1}{n\ln 2 \times \epsilon n^{\epsilon-1}} = 0$
4. Since $\lim_{n\rightarrow\infty}\frac{f(n)}{g(n)} = 0$, $f(n)=o(g(n)) \therefore$ **contradiction**
    - ie. $f(n)$ is asymptotically smaller than $g(n)$ which contradicts our assumption at line (2)
5. Thus, $f(n)$ is not polynomially larger than $g(n)$ and case 3 does not apply

#### Steps to Take

$$T(n) = a\cdot T \left( \frac{n}{b} \right) + f(n)$$

**Case 1**: $f(n)$ is **asymptotically smaller** than $n^{\log_ba}$, ie. $\lim_{n\rightarrow\infty}\frac{f(n)}{n^{\log_ba}}<\infty$

1. Let $g(n) = n^{\log_ba-\epsilon}$, for some constant $\epsilon > 0$
2. Check if $f(n)$ is **polynomially smaller** than $n^{\log_ba}$, ie. $f(n)=O(g(n))$
3. $\lim_{n\rightarrow\infty}\frac{f(n)}{g(n)} < \infty \implies f(n)=O(g(n))$
4. $f(n)=O(g(n)) \implies T(n) = \Theta(n^{\log_ba})$
5. If $f(n)$ is not polynomially smaller, recurrence cannot be solved by Master Theorem

**Case 2**: $f(n)$ is **asymptotically equal** to $n^{\log_ba}\lg^kn$ for some $k\geq0$, ie. $0<\lim_{n\rightarrow\infty}\frac{f(n)}{n^{\log_ba}\lg^kn}<\infty$

1. $T(n) = \Theta(n^{\log_ba}\lg^{k+1} n)$

**Case 3**: $f(n)$ is **asymptotically larger** than $n^{\log_ba}$, ie. $\lim_{n\rightarrow\infty}\frac{f(n)}{n^{\log_ba}}>0$

1. Let $g(n) = n^{\log_ba+\epsilon}$, for some constant $\epsilon > 0$
2. Check if $f(n)$ is **polynomially larger** than $n^{\log_ba}$, ie. $f(n)=\Omega(g(n))$
3. $\lim_{n\rightarrow\infty}\frac{f(n)}{g(n)} > 0 \implies f(n=\Omega(g(n))$
4. Check if **regularity condition** holds, ie. $af(\frac{n}{b}) \leq cf(n)$ for some $c < 1$ and for all sufficiently large $n$
5. $f(n)=\Omega(g(n)) \wedge af(\frac{n}{b}) \leq cf(n) \implies T(n) = \Theta(f(n))$
6. If either condition not satisfied, recurrence cannot be solved by Master Theorem

### Substitution Technique

$$T(n)=2T\left(\frac{n}{2}\right)+\Theta(n)$$

- **Guess**: $T(n)=O(n\lg n)$
  - ie. $\exist c>0, T(n)\leq cn\lg n$
- **Assume**: that our guess is true for $\forall m<n$
  - ie. $\exist c>0, T(m)\leq cm\lg m$
- **Prove**: $T(n)\leq cn\lg n$ under the above assumption
  1. Let $m=\frac{n}{2}$ (since $\frac{n}{2}$ is clearly less than $n$, our assumption holds)
  2. $T(\frac{n}{2}) \leq c\frac{n}{2}\lg\frac{n}{2}$
  3. $2T(\frac{n}{2}) \leq cn\lg\frac{n}{2}$
  4. Substituting into original recurrence: $T(n)\leq cn\lg\frac{n}{2}+n$
  5. $T(n)\leq cn(\lg n - \lg 2)+an$
  6. $T(n)\leq cn(\lg n - 1)+an$
  7. $T(n)\leq cn\lg n - cn +an$
  8. $T(n)\leq cn\lg n - (c-a)n \leq cn\lg n$ for any sufficiently large $c>a$
  9. Thus, we have proven our statement, initial guess must be true

#### Negative Example (Wrong Guess)

- **Guess**: $T(n)=O(n)$
- **Assume**: that our guess is true for $\forall m<n$
- **Prove**: $T(n)\leq cn$ under the above assumption
  1. Let $m=\frac{n}{2}$ (since $\frac{n}{2}$ is clearly less than $n$, our assumption holds)
  2. $T(\frac{n}{2}) \leq c\frac{n}{2}$
  3. $2T(\frac{n}{2}) \leq cn$
  4. Substituting into original recurrence: $T(n)\leq cn+an$
  5. However, $cn+an \not\leq cn$, thus we have failed to prove our statement

## Divide & Conquer

1. Divide the problem into a number of subproblems that are smaller instances of the same problem
2. Conquer the subproblems by solving them recursively. If the subproblem sizes are small enough, just solve the subproblems in a straightforward manner
3. Combine the solutions to the subproblems into the solution for the original problem

### Technique

1. Make 2 subproblems become just 1 subproblem
    - eg. from $T(n)=2T(\frac{n}{2})+1 = \Theta(n)$ to $T(n)=T(\frac{n}{2})+1 = \Theta(\log n)$

### Matrix Multiplication

Given $A=[a_{ij}], B =[b_{ij}]$ and $C=[c_{ij}]=A\times B$,
$c_{ij}=\sum_{k=1}^na_{ik}\times b_{kj}$


## Dynamic Programming

### Example: Computing Fibonacci

#### Recursive Approach

```py
fib(n):
    if n <= 2: f = 1
    else: f = fib(n - 1) + fib(n - 2)
    return f
```

Let $T(n)$ be the number of instructions executed by `fib(n)`.

$$
\begin{aligned}
T(n) &= T(n-1) + T(n-2) + \Theta(1) \text{ for } n>1\\
  &\geq 2T(n-2) + \Theta(1) \text{ since $T(n-1) \geq T(n-2)$}\\
  &\geq 2^{\frac{n}{2}}
\end{aligned}
$$

Thus, $T(n)$ is at least exponential in $n$. The exact bound is $\phi^n$ (where $\phi$ is the golden ratio).

> Intuition: multiplying by $2$ each time, and subtracting by $2$ each time. We can subtract $2$ from $n$ exactly $\frac{n}{2}$ times. Thus, $2^\frac{n}{2}$.

<img src="./assets/fib-recursion-tree.png" height="250">

- During execution of recursive `fib`, observe that the whole recursion tree is visited
- Number of instructions must be at least number of nodes in this recursion tree

#### Memoized DP Approach

```py
memo = {} # initialise empty dictionary
fib(n):
    if n in memo: return memo[n]
    if n <= 2: f = 1
    else: f = fib(n - 1) + fib(n - 2)
    memo[n] = f
    return f
```

Notice that lines $4$ and $5$ are exactly the same as in the recursive approach.

#### Iterative DP Approach

```py
fib = {} # initialise empty dictionary
for k in range(1, k + 1):
    if k <= 2: f = 1
    else: f = fib[k - 1] + fib[k - 2]
    fib[k] = f
return fib[n]
```

Notice again that lines $3$ and $4$ are exactly the same as in the recursive approach.

Let $T(n)$ be the number of instructions executed,

$$
\begin{aligned}
T(n) &= T(n-1) + \Theta(1)\\
  &= \Theta(n)
\end{aligned}
$$

<img src="./assets/fib-recursion-tree-memo.png" height="250">

### Example: Longest Common Subsequence (LCS)

Given: 2 sequences $A[1..n]$ and $B[1..m]$

Aim: compute a (not "the", since there can be multiple) longest sequence $C$ such that $C$ is subsequence of both $A$ and $B$

$A :$ **H**I**E**ROG**L**YPHO**LO**GY  
$B :$ MIC**H**A**EL**ANGE**LO**  
$LCS(A,B) = C :$ HELLO

#### Recursive Solution

Given: 2 sequences $A[1..n]$ and $B[1..m]$

Base Cases

$$LCS(i,0)=\emptyset$$

$$LCS(0,j)=\emptyset$$

General Case

$$
\begin{aligned}
a_n &= b_m \implies LCS(n,m) = LCS(n-1,m-1) :: a_n\\
a_n &\neq b_m \implies LCS(n,m) = \text{Max}[LCS(n-1,m), LCS(n,m-1)]
\end{aligned}
$$

Thus total time $= O(m2^n)$.

#### Recursive Solution for Length of LCS

Given: 2 sequences $A[1..n]$ and $B[1..m]$

Aim: find length of the LCS of $A$ and $B$, $L(i,j)$ recursively.

Base Cases

$$L(i,0)=0$$

$$L(0,j)=0$$

General Case

$$
\begin{aligned}
a_n &= b_m \implies L(n,m) = L(n-1,m-1) + 1\\
a_n &\neq b_m \implies L(n,m) = \text{Max}[L(n-1,m), L(n,m-1)]
\end{aligned}
$$

```py
L(n, m):
    # base case
    if n == 0 or m == 0:
        return 0
    # general case
    if A[n] == B[m]:
        return L(n - 1, m - 1) + 1
    else:
        return Max(L(n - 1, m), L(n, m - 1))
```

$$
\begin{aligned}
T(n,m) &= T(n-1,m) + T(n,m-1) + c\\
  &\geq \binom{n+m}{n}\\
  &> 2^n \text{ (assuming $m \approx n$)}
\end{aligned}
$$

This time complexity is no different from the brute force approach as we are solving **overlapping subproblems** multiple times:

<img src="./assets/length-lcs-recursion-tree.png" height="300">

Thus, we can apply dynamic programming approach:

```py
L(n, m):
    # dp is a 2D array, n is col, m is row
    dp = Array[n, m]
    for i = 0 to n:
        dp[i, 0] = 0 # init bottom row of dp to 0
    for i = 0 to m:
        dp[0, i] = 0 # init left col of dp to 0
    for j = 1 to m:
        for i = 1 to n:
            # general case in recursion
            if A[i] == B[j]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = Max(dp[i - 1, j], dp[i, j - 1])
```

<img src="./assets/length-lcs-table.png" height="250">

- Total time $= O(nm)$
- Total Space $= O(nm)$

### Optimal Substructure

A problem exhibits optimal substructure if an optimal solution to the problem contains within it optimal solutions to subproblems. For example, solving problem $S_n$ requires solving $S_{n-1}$ first, and subsequently $S_{n-2},\dots$.

To prove that an optimal substructure exists:

1. Show that a solution to the problem consists of making a choice, and making this choice leaves one or more subproblems to be solved
2. Assume that the choice that leads to an optimal solution is given (do not concern yourself yet with how to determine this choice, just assume it has been given)
3. Given this choice, determine which subproblems ensue and how to best characterise the resulting space of subproblems
4. Show that the solutions to the subproblems used within an optimal solution to the problem must themselves be optimal by using a "cut-and-paste" technique:
   1. Assume each of the subproblem solution is not optimal
   2. By "cutting out" the non-optimal solution to the subproblems and "pasting in" the optimal one, we can get a better solution to the original problem
   3. This contradicts our assumption of line 2 that we already have an optiaml solution  
   4. Thus, assumption that subproblem solution is not optimal must be incorrect; the subproblem solutions must themselves be optimal

### Designing Dynamic Algorithms

1. Find the **optimal substructure**
   - Express the solutions recursively
2. Find the **overlapping subproblems**
   - Observe that there are only polynomial number of distinct subproblems
   - However, the recursive algorithm takes non-polynomial/exponential time because it is solving the same subproblems multiple times
3. Thus, we compute the recursive solution **iteratively in a bottom-up fashion (or top-down with memoization)**

### Example: Longest Palindrome Subsequence

Use the LCS algorithm to give an $O(n^2)$ time algorithm to find the longest palindrom subsequence of an input string. For example, for `character`, the output should be `carac`.

Solution: run LCS algorithm on both the input string, and the reverse of the input string. Thus, running time $= O(n \cdot n)$

### Example: Knapsack Problem

Given: weight-value pairs $(w_1,v_1),(w_2,v_2),\dots,(w_n,v_n)$, and target weight $W$

Ouput: a subset $S \subseteq \{1,2,\dots,n\}$ that maximises value $\sum_{i \in S} v_i$ such that $\sum_{i \in S}w_i \leq W$

Observe that there are $2^n$ many subsets, and the brute force approach will take $O(2^n)$ time.

**Find the optimal substructure**:

Let $m[i,j]$ be the maximum value that can be obtained by using a subset of items in $\{1,2,\dots,i\}$ with total weight no more than $j$. Thus, our objective is to find $m[n,w]$

Base case: $i = 0$ or $j = 0$

$$m[i,j] = 0$$

Case 1: item $i$ is taken (ie. $w_i \leq j$)

$$
m[i,j] = \text{Max}\{ m[i-1,j-w_i] + v_i, m[i-1,j] \}
$$

Case 2: item $i$ is not taken (ie. $w_i > j$)

$$
m[i,j] = m[i-1,j]
$$

**Compute the solution iteratively in a bottom-up fashion**:

```py
knapsack(v, w, W):
    m = Array[n, W] # m is a n*W array
    for j = 0 to W:
        m[o, j] = 0
    for i = 1 to n:
        m[i, 0] = 0

    for i = 1 to n:
        for j = 0 to W:
            if j >= w[i]:
                m[i, j] = Max(m[i - 1, j - w[i]] + v[i], m[i - 1, j])
            else:
                m[i, j] = m[i - 1, j]

    return m[n, W]
```

- Time taken to fill up a table entry $= O(1)$
- Total time $= O(nW)$
- Total space $= O(nW)$

### Example: Single Source Shortest Path

Given: directed graph $G=(V,E)$ with with some edge weight $\omega$, and a soruce vertex $s \in V$.

Notations:

- $n=|V|$, $m=|E|$
- $\delta(u,v)$: distance from $u$ to $v$
- $P(u,v)$: shortest path from $u$ to $v$

Objectives:

1. Compute $\delta(s,v)$ for all $v\in V\backslash\{s\}$
1. Compute $P(s,v)$ for all $v\in V\backslash\{s\}$

#### Dijkstra's Algorithm

If all edge weights $\omega$ are **positive**, Dijkstra's can solve this in $O(m + n\log n)$ time.

Two crucial facts exploited:

1. The nearest neighbour is also the vertex nearest to $s$
2. Optimal subpath property: given a shortest path $P(s,v)$ that passes through $u$, $P(s,u)$ and $P(u,v)$ are themselves shortest paths

However, if there are **negative** edges weights, the properties are violated:

1. The nearest neighbour is not necessarily the vertex nearest to $s$. There can exist some other path with negative edge weights that are not the direct neighbour of $s$
2. $\delta(s,y)=5$ by going $s \rightarrow u \rightarrow v \rightarrow x \rightarrow y$, but $\delta(s,v)=40$ (green path) and is not the subpath of $P(s,y)$, given by $s \rightarrow u \rightarrow v$, which will yield distance of $60$
<img src="./assets/sssp-violation.png" width="270">

#### Bellman-Ford Algorithm

**Theorem**: all shortest paths in $G$ (with negative edge weights) possess optimal substructure property **if there are no negative cycles**.

Let $L(v,i)$ denote the weight of the shortest path from $s \rightsquigarrow v$ having at most $i$ edges. Limit the number of edges because there can exist some other path which takes $>i$ edges which is shorter.

SSSP can now be converted to compute $L(v, n-1)$ for each $v$. Since there are no cycles, each vertex can only be visited at most once, and thus $n-1$ edges will be used.

**Recursive formulation**:

1. If $L(v,i)$ has $<i$ edges: $L(v,i)=L(v,i-1)$
2. If $L(v,i)$ has exactly $i$ edges: $L(v,i)=min_{(x,v)\in E}(L(x,i-1)+\omega(x,v))$

$Bellman\text{-}Ford$

1. For each $v \in V \backslash\{s\}$:
   1. If $(s,v)\in E$: $L[v,1] = \omega(s,v)$
   2. Else: $L[v,1] = \infty$
2. $L[s,1]=0$
3. For $i=2$ to $n-1$:
   1. For each $v\in V$:
      1. $L[v,i]=L[v,i-1]$
      2. For each $(x,v)\in E$:
         1. $L[v,i] = min(L[v,i],L[x,i-1]+\omega(x,v)$

**Time**: $O(mn)$, in the innermost for loop (step $3.1.2$), we are just using $deg(v)$ time. Since we do this for each $v$, step $3.1$ will take overall $\sum_{v \in V} deg(v) = 2m$ time. Another way to look at it is that in step $3.1$, we are simply relaxing every edge once.  
**Space**: $O(n^2)$

## Greedy Algorithms

A greedy algorithm obtains an optimal solution to a problem by making a sequence of choices. At each decision point, the algorithm makes a choice that **seems best at the moment** (ie. the local best). This heuristic strategy **does not** always produce an optimal solution.

### Example: Longest Increasing Subsequence (LIS)

Example: $10,3,7,2,4,6,11,8,5,9$  
Output: $2,4,6,8,9$

1. Sort the $n$ numbers to form sorted list, $O(n\log n)$
2. Use DP apporach to find LCS between sorted and original sequence, $O(n^2)$

### Example: Solitaire (LIS via Patience Sorting)

You are given some shuffled cards. Cards must be drawn individually and placed in a pile such that the order of cards in the pile is in decreasing rank. Find the least number of piles required.

Greedy strategy: always try to place a card on the leftmost pile. In no such pile exists, create a new pile on the right side.

Observe that:

- Cards in each pile are in decreasing order
- Any increasing subsequence of the deck contains at most one card from each pile
  - Piles are in decreasing order and any card that is in the top of a pile has come after cards at the bottom
  - Thus, length of LIS $\leq$ minimum number of piles

We know that:

1. Minimum number of piles $\leq$ number of piles in greedy strategy
2. Number of piles in greedy strategy $\leq$ length of LIS

Combining these and the observation above, minimum number of piles $\leq$ number of piles in greedy strategy $\leq$ length of LIS $\leq$ minimum number of piles. As such, minimum number of piles $=$ number of piles in greedy strategy $=$ length of LIS $=$ minimum number of piles

#### Algorithm and Analysis

1. Use a stack to implement each pile
   - At most $n$ stacks, $1$ for each of the $n$ cards
2. Place card on the leftmost allowed pile, else create a new pile
   - Since the top cards of each stack are in sorted order, binary search can be used, $O(n\log n)$

Time: $O(n\log n)$

### Designing Greedy Algorithms

1. Cast the optimisation problem as one in which we make a choice and are left with one subproblem to solve
2. Prove that there is always an optimal solution to the original problem that makes the greedy choice, so that the greedy choice is always safe
3. Demonstrate optimal substructure by showing that, having made the greedy choice, what remains is a subproblem with the property that if we combine an optimal solution to the subproblem with the greedy choice we have made, we arrive at an optimal solution to the original problem

> Beneath every greedy algorithm, there is almost always a more cumbersome dynamic-programming solution

### Proving Correctness of Greedy Algorithms

Proof by contradiction:

1. Assume that there exists an optimal solution that is better than our greedy solution
2. Show that some parts of the optimal solution is different from greedy solution (assuming optimal is better, some steps taken by optimal *must* be different from greedy)
3. Show that this different steps will result in optimal solution being worse than greedy and hence contradicting our assumption that optimal is better than greedy

Refer here for detailed proof of Fractional Knapsack and Activity Scheduling: <https://www2.cs.duke.edu/courses/fall17/compsci330/lecture6note.pdf>

> Alternatively, proof by induction and other methods learnt can be used too

### Example: Huffman Code

Given: alphabet set $A : \{ a_1,a_2,\dots,a_n \}$

How many bits are needed to encode a text file with $m$ characters?

Trivial answer: $m\lceil\log_2n\rceil$ bits

> Can we use fewer bits to store set $A$?  
> No.
>
> Can we use fewer bits to store a text file?  
> Yes, due to the huge variation in the frequency of alphabets in a text.

Intuitively, use shorter bit encodings for more frequent alphabets and longer bit encodings for less frequent alphabets.

<img src="./assets/variable-length-encoding.png" height="250"/>

A coding $\gamma(A)$ is called **Prefix Coding** if there does not exists $x,y \in A$ such that $\gamma(x)$ is a prefix of $\gamma(y)$.

**Problem**: given a set of $A$ of $n$ alphabets and their frequencies, compute a coding $\gamma$ such that:

1. $\gamma$ is a prefix coding
2. $ABL(\gamma)$ is minimum

Solution: labelled binary tree

<img src="./assets/labelled-binary-tree.png" height="250">

> Similarly, given a prefix code, we can also work backwards to find the labelled binary tree.

**Theorem**: for each prefix code of a set $A$ of $n$ alphabets there exists a binary tree $T$ on $n$ leaves such that:

1. There is a bijective mapping between the alphabets and the leaves
2. The label of a path from root to a leaf node corresponds to the prefix code of the corresponding alphabet

Thus, we can express the $ABL$ of $\gamma$ in terms of its binary tree $T$,

$$
\begin{aligned}
ABL(\gamma) &= \sum_{x\in A}f(x)\cdot |\gamma(x)|\\
  &= \sum_{x\in A} f(x) \cdot \text{depth}_T(x)
\end{aligned}
$$

In other words, if we are able to find the smallest correct labelled binary tree, we would also solve the problem of optimal prefix codes.

**Lemma**: the binary tree corresponding to the optimal prefix coding must be a **full** binary tree (every internal node must have degree exactly $2$). Thus, using this lemma, any full binary tree will guarantee that $ABL$ is minimised.

Intuitively, more frequent alphabets should be closer to the root.

Let $A = a_1,a_2,\dots,a_n$ be $n$ alphabets in increasing order of frequencies.  
Let $A = a_3,\dots,a',\dots,a_n$ be $n-1$ alphabets in increasing order of frequencies with $f(a') = f(a_1) + f(a_2)$.  

Thus, $Optimum_{ABL}(A) = Optimum_{ABL}(A') + f(a_1) + f(a_2)$

#### Algorithm

$OPT(A)$

1. If $|A| = 2$
   1. return $(a_1,a_2)$
2. Else,
   1. Let $a_1$ and $a_2$ be the two alphabets with least frequencies [initial $O(n\log n)$ time to sort]
   2. Remove $a_1$ and $a_2$ from $A$
   3. Create a new alphabet $a'$, where $f(a') = f(a_1) + f(a_2)$
   4. Insert $a'$ into $A$ to form $A'$ [do binary search to insert, $O(\log n)$]
   5. Build optimum tree $T = OPT(A')$
   6. Replace node $a'$ in $T$ by $(a_1 \leftarrow^0 o \rightarrow^1 a_2)$
   7. return $T$

Time: $O(n \log n)$

#### Proof of Correctness

Given $OPT_{ABL}(A')$,

$$
\begin{aligned}
ABL(A) &= OPT_{ABL}(A') + f(a_1) + f(a_2)\\
OPT_{ABL}(A) &\leq OPT_{ABL}(A') + f(a_1) + f(a_2) \Rightarrow (1)
\end{aligned}
$$

Given $OPT_{ABL}(A)$,

$$
\begin{aligned}
ABL(A') &= OPT_{ABL}(A) - f(a_1) - f(a_2)\\
OPT_{ABL}(A') &\leq OPT_{ABL}(A) - f(a_1) - f(a_2)\\
OPT_{ABL}(A) &\geq OPT_{ABL}(A') + f(a_1) + f(a_2) \Rightarrow (2)
\end{aligned}
$$

Thus, by equation $(1)$ and $(2)$, $OPT_{ABL}(A) = OPT_{ABL}(A') + f(a_1) + f(a_2)$

### Example: Fractional Knapsack

Given input weight-value pairs $(w_1,v_1),(w_2,v_2),\dots,(w_n,v_n)$ and maximum weight $W$, find the maximum possible sum of values $V$ that could be achieved such that $W$ is not exceeded.

**Optimal Substructure**: If we remove $w$ kg of an item $j$ from the optimal knapsack, then the remaining load must be the optiam knapsack weighing at most $W-w$ kg that one can take from $n-1$ original items and $w_j-w$ kg of item $j$.

**Greedy Choice Property**: an optimal solution is one that has the highest total value density (highest value per kg). Since we are always adding as much of the highest value density we can, we are going to end up with the highest total value density. Suppose otherwise that we had some other solution that used some amount of the lower value density object. However, we could easily substitute these lower value density objects with some of the higher value density objects meaning our original solution coud not have been optimal.

#### Algorithm

1. Sort items by decreasing order of the value density: $O(n\log n)$
2. If target weight $W$ is not yet met, recursively take as much of the highest value density objects: $O(n)$

Total time: $O(n\log n)$
