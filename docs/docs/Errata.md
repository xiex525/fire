---
title: Errata
permalink: /errata/
nav_order: 1.1
usemathjax: true
---


# Errata for *Navigating the Factor Zoo: The Science of Quantitative Investing*

$$x$$ math at the start of the document

Nested: $$M = \text{while e^2 $e^2$ do $c^2$ end}$$

**Numbered equations**

$$
\begin{equation}
\int_0^x \sin(x) dx
\label{eq:test}
\end{equation}
$$



$$R V_{t}^{+} \rightarrow \mathop{\phantom{\,}}_{t-1}^{t} \sigma_{s}^{2} ds+\sum_{t-1 \leq \tau \leq t} J_{\tau}^{2} J_{\tau}>0 $$
---
## Overview

This document captures all **verified printing and content errors** identified in *_Navigating the Factor Zoo: The Science of Quantitative Investing_*. It is maintained in the Fire Institute GitHub repository (https://github.com/fire-institute/fire) under `docs/docs/errata.md`.

### Structure of Entries
Each erratum follows this format:

| Field              | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| **Anchor**         | Unique Markdown heading used as the link target.                   |
| **Original**       | Verbatim the incorrect text, caption, or equation.                 |
| **Correction**     | The accurate replacement text, caption, or equation.               |
| **Note**           | (Optional) Additional context or explanation.                      |

---

### Submitting a New Error Report
To contribute:

1. **Search** existing GitHub issues to avoid duplicates.  
2. **Open a new issue** with the title:
   ```
   [Errata] Page <number> – brief description
   ```
3. **Fill in the template** in the issue body:
   ```markdown
   **Page**:  
   **Section or Heading**:  
   **Original**:  
   **Correction**:  
   ```
4. A maintainer will review, label it **confirmed**, and then add it here.

---

## Edition Index

* [First Edition — Routledge (Hardcover & Paperback)](#first-edition-routledge-hardcover--paperback)
  * [Page 66 – Equation 3.19](#page-66-equation-3-19)

---

## First Edition — Routledge (Hardcover & Paperback) <a name="first-edition-routledge-hardcover--paperback"></a>

- **Publisher**: Routledge  
- **Publication Date**: November 20, 2024 (Hardcover) / December 9, 2024 (Paperback)  
- **Formats**: Hardcover (296 pp.) / Paperback (310 pp.)  
- **ISBN-10**: 1032768436 (HC) / 103276841X (PB)  
- **ISBN-13**: 978-1032768434 (HC) / 978-1032768410 (PB)  

### Page 66 – Equation 3.19 <a name="page-66-equation-3-19"></a>
**Original**  

$\rightarrow \mathbb{l}  \mathcal{l} $

> In the limit of $n \rightarrow \infty$，$R V_{t}^{+} \rightarrow \text{ }_{t-1}^{t} \sigma_{s}^{2} ds+\sum_{t-1 \leq \tau \leq t} J_{\tau}^{2} J_{\tau}>0$
> ```math
> \text{In the limit of } n \rightarrow \infty \text{ , } R V_{t}^{+} \rightarrow \text{ }_{t-1}^{t} \sigma_{s}^{2} ds+\sum_{t-1 \leq \tau \leq t} J_{\tau}^{2} J_{\tau}>0  \text{ , }  R V_{t}^{-} \rightarrow \int_{t-1}^{t} \sigma_{s}^{2} d s+\sum_{t-1 \leq \tau \leq t} J_{\tau}^{2} J_{\tau} \text{  and, }
> ```
> ```math
> S J_{t}=\sum_{t-1 \leq \tau \leq t} J_{\tau}^{2} J_{\tau}>0-\sum_{t-1 \leq \tau \leq t} J_{\tau}^{2} J_{\tau} 0
> ```

**Correction**  

> ```math
> \text{In the limit of } n\to \infty \text{ , } RV_t^+ \to \int _{t- 1}^t\sigma _s^2ds+ \Sigma_{t- 1\leq \tau \leq t}J_\tau^2\mathbb{l} _{J_\tau > 0} \text{ , } RV_t^- \to \int_{t- 1}^t\sigma_s^2ds + \Sigma_{t-1\leq\tau\leq t} J_{\tau}^{2}]_{J_{\tau<0}} \text{, and,}
> ```
>  ```math
> SJ_t=\sum_{t-1\leq\tau\leq t}J_\tau^2\mathbb{I}_{J_\tau>0}-\sum_{t-1\leq\tau\leq t}J_\tau^2\mathbb{I}_{J_\tau<0}.
> ```



 


**Note**  
> The summations now correctly use indicator functions to separate positive and negative jumps. The continuous integral term $\int_{t-1}^t \sigma_s^2 ds$ was previously missing the integral symbol and has now been properly reinstated.

