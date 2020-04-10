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
### "Testing": Returns a matrix with entries $$A_{ij} ~ \int_{[x_j x_{j+1}]} f(x)P_{i-1}dx$$
```py
def testing(f, xx, k):  # returns a (k+1)x(No. of elements) matri
    n_el = np.size(xx) - 1
    hh = 0.5 * (xx[:, 1:n_el + 1] - xx[:, 0:n_el])
    t, wts = gaussian_quad(k + 2)
    p_si, _ = polynomial_basis(k, t)
    aa = np.transpose(wts) * p_si
    aa = hh * (np.transpose(aa) @ f(quad_points(xx, t)))
    return aa
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
