<!-- This can be hide, original code: http://zjuwhw.github.io/2017/06/04/MathJax.html --> 
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A High Order Finite Element Code for $$H^1(a,b)$$
## IPython version

This is a "simple" implementation for a high order FEM  for  $$H^1(a,b)$$. We use Lobatto functions   

### Libraries
```py
import numpy as np                       # package for scientific computing
from scipy.linalg import eigh            # eigenvalues and eigenvectors for symmetric matrices
from numpy.polynomial import Legendre    # Legendre : Legendre polynomials
from scipy.sparse import csc_matrix      # csc_matrix : similar to sparse() in Matlab
from scipy.sparse.linalg import spsolve  # spsolve : solves sparse linear systems AX = B
import matplotlib.pyplot as plt          # package for plotting
import cProfile                          # provide deterministic profiling
```
### Basis in the reference element $$[-1,1]$$.
```py
def polynomial_basis(k, t):  # Lobatto functions of degree k at the points t
    # Last Modified Friday December 21 2018
    m = len(t)                                                             # length of t
    leg = np.zeros([m, k + 1])
    for kind in range(0, k+1):
        leg[:, kind] = np.sqrt((2*kind+1) / 2) * Legendre.basis(kind)(t)   # Legendre Normalized poly at t
    p = leg * np.sqrt(2.0 / (2 * np.arange(k + 1) + 1))
    psi = np.zeros([m, k+1])
    # Lobatto polynomial functions
    psi[:, 0] = 0.5 * (1 - t)
    psi[:, 1] = 0.5 * (1 + t)
    correction = np.sqrt(1 / (2 * (2 * np.arange(2, k + 1) - 1)))
    psi[:, 2:k + 1] = correction * (p[:, 2:k+1] - p[:, 0:k-1])
    # derivative Lobatto polynomial functions
    psi_p = np.zeros([m, k + 1])
    psi_p[:, 0] = -0.5
    psi_p[:, 1] = 0.5
    psi_p[:, 2:k+1] = leg[:, 1:k]
    return psi, psi_p
```
### Quadrature rule
```py
def quad_points(x, tt):  # integration pts in the physical elements; t quadrature points in (-1,1), x is the partition
    # Last Modified Wednesday December 18 2018
    n = np.size(x)
    nqd = len(tt)
    tt = tt.reshape((1, nqd))
    h = 0.5 * (x[:, 1:n]-x[:, 0:n-1])
    aa = ((np.transpose(tt)+1)*h)+x[:, 0:n-1]
    return aa
```
### "Testing": Returns a matrix with entries $$A_{ij}  \approx \int_{[x_j x_{j+1}]} f(x)P_{i-1}dx$$
```py
def testing(f, xx, k):  # returns a (k+1)x(No. of elements) matrix
    n_el = np.size(xx) - 1
    hh = 0.5 * (xx[:, 1:n_el + 1] - xx[:, 0:n_el])
    t, wts = gaussian_quad(k + 2)
    p_si, _ = polynomial_basis(k, t)
    aa = np.transpose(wts) * p_si
    aa = hh * (np.transpose(aa) @ f(quad_points(xx, t)))
    return aa
```
### Mass Matrix ( $$d\mu =\rho(x)dx$$)
```py
def mass_matrix(rho, xx, k):    # Output: A := (k+1) x (k +1) x (No. of elements) matrix (Numpy order is different)
                                # x must be an (1, n) array,
                                # rho: vectorized function, the output must have same shape than input.
                                # k: poly degree
    n_el = np.size(xx) - 1
    hh = 0.5 * (xx[:, 1:n_el+1]-xx[:, 0:n_el])
    n_qd = int(np.ceil(1.5*k+0.5))
    points, weights = gaussian_quad(n_qd)
    p_si, _ = polynomial_basis(k, points)
    rr = rho(quad_points(xx, points))
    rr = rr * hh                                    # Column-wise multiplication
    aa = np.zeros((k+1, (k+1)*n_el))
    for q in np.arange(n_qd, dtype=np.uint32):
        aa = aa + np.kron(rr[q, :], weights[:, q] * np.outer(p_si[q, :], p_si[q, :]))
    aa = np.transpose(aa)
    aa = aa.reshape(n_el, k+1, k+1, order='C')
    return aa
```
### Stiffness Matrix ( $$d\mu =c(x)dx$$)
```py
def stiffness_matrix(c, xx, k):     # Output: A := (k+1) x (k +1) x (No. of elements) matrix (Numpy order is different)
                                    # x must be an (1, n) array,
                                    # c: vectorized function, the output must have same shape than input.
                                    # k: poly degree
    n_el = np.size(xx) - 1
    hh = 0.5 * (xx[:, 1:n_el + 1] - xx[:, 0:n_el])
    n_qd = int(np.ceil(1.5 * k))                     # np.ceil returns a float64
    points, weights = gaussian_quad(n_qd)
    _, p_sip = polynomial_basis(k, points)
    cc = c(quad_points(xx, points))
    cc = cc * (1/hh)                                 # Column-wise multiplication
    aa = np.zeros((k + 1, (k + 1) * n_el))
    for q in np.arange(n_qd, dtype=np.uint32):
        aa = aa + np.kron(cc[q, :], weights[:, q] * np.outer(p_sip[q, :], p_sip[q, :]))
    aa = np.transpose(aa)
    aa = aa.reshape(n_el, k + 1, k + 1, order='C')
    return aa
```
### Degrees of freedom (table)
```py
def dof(n_elt, k):   # local-to-global operator (k+1) x Nelt matrix.
    # Last Modified Tuesday December 18 2018
    l2g = np.ones([k+1, n_elt], dtype=np.uint32)   # zero-initialization, it works until 4.294.967.294 dof
    g = np.arange(1, k + 2, dtype=np.uint32)        # g = [1 2 3 ··· k k+1]
    g = np.roll(g, 1)                               # g = [k+1 1 2 3 ··· k]
    g[0] = 1                                        # g = [1 1 2 3 ··· k]
    g[1] = k + 1                                    # g = [1 k+1 2 3 ··· k]
    g = np.reshape(g, (k+1, 1))
    col = np.arange(0, n_elt)
    l2g = ((k * l2g) * col)+g
    return l2g-1
```
### Assembly Mass Matrix or Stiffness Matrix
```py
def assemble(w):  # Assembly of W, W is (k+1) x (k+1) x Nelt (Numpy shape (Nelt,k+1,k+1))
    dim = w.shape
    n_el = dim[0]
    k = dim[1] - 1
    l2g = dof(n_el, k).flatten('F')            # l2g(:)
    l2g = np.reshape(l2g, ((k+1) * n_el, 1))   # col vector N x 1
    cols = np.tile(l2g, k+1)
    cols = np.reshape(cols, (n_el, k+1, k+1), order='C')
    cols = np.transpose(cols, (0, 1, 2))
    cols = np.moveaxis(cols, 1, 2)
    rows = np.moveaxis(cols, 1, 2)
    w = w.flatten('F')                         # w = w(:)
    cols = cols.flatten('F')
    rows = rows.flatten('F')
    ww = csc_matrix((w, (rows, cols)), shape=(n_el*k+1, n_el*k+1))
    return ww
```
### Assembly load vector
```py
def vector_assemble(w):  # Assembly of W, load vector Numpy (k+1,n_el) : similar to accumarray Matlab
    dim = w.shape
    n_el = dim[1]               # dim(end)
    k = dim[0]-1
    l2g = dof(n_el, k).flatten('F')
    w = w.flatten('F')
    ww = np.zeros((k*n_el+1, 1)).flatten('F')     # k*n_el+1 = dimension space (system size)
    np.add.at(ww, l2g, w)
    return ww
```
### Solver for $$ (cf(x)u'(x))'+\rho(x)u(x)=f(x)$$ plus boundary conditions. D: Dirichlet, N: Neumann
```py
def fem_1d(xx, k, cf, rho, bc, f, bval):
    n_el = np.size(xx) - 1
    n_dof = (n_el * k) + 1
    free = np.arange(n_dof)
    ss = assemble(stiffness_matrix(cf, xx, k) + mass_matrix(rho, xx, k))
    b = vector_assemble(testing(f, xx, k))
    if bc[0] == "N":
        b[0] = b[0] + bval[0]
    if bc[1] == "N":
        b[1] = b[1] + bval[1]
    uh = np.zeros((n_dof, 1)).flatten('F')
    dir = []
    if bc[0] == "D":
        free = free[1:n_dof]
        dir = [0]
        uh[0] = bval[0]
    if bc[1] == "D":
        free = free[0:-1]
        dir = dir+[n_dof-1]
        uh[n_dof-1] = bval[1]
    b = np.transpose(b) - (ss[:, dir] @ uh[dir]).flatten('F')
    uh[free] = spsolve(ss[free, :][:, free], b[free])
    return uh
```



You can use the [editor on GitHub](https://github.com/Hugo-Diaz-N/Hugo-Diaz-N.github.io/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
Ruby Code
```ruby
# ...ruby code
print "Hello, World!\n"
```




For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

[Languages supported by Jekyll](https://simpleit.rocks/ruby/jekyll/what-are-the-supported-language-highlighters-in-jekyll/)  

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Hugo-Diaz-N/Hugo-Diaz-N.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
