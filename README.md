# 📊 Linear Regression from Scratch with Least Squares & Gradient Descent

Welcome to this educational project that demonstrates how to solve a **linear regression** problem using pure **Python** and **NumPy** with no machine learning libraries. The solution is built from scratch based on core mathematical concepts like:

- 🔢 **Least Squares Error**
- 🔁 **Gradient Descent Algorithm**

---
## 🏡 Boston home price prediction

I gathered some data about the Boston home prices, rooms, and area.
This model cannot be very accurate in predicting house prices because it is a simple scratch model and advanced machine learning libraries were not used to build the model. The focus of this project is on the basic concepts of linear regression.

---
### 📁 Dataset
<p>
    <img src="assets/data.png" width="400" alt="dataset" />
<p/>


```python
# Sample data

equations = [
    [4, 130, 600000],
    [4, 185, 1900000],
    [2, 90, 303000],
    [3, 135, 780000],
    [2, 100, 925000],
    [4, 120, 900000],
    [3, 222, 950000],
    [3, 200, 910000],
    [3, 154, 885000],
    [2, 66, 300000]
]

# Convert to NumPy arrays
A = np.array([[e[0], e[1]] for e in equations], dtype=float)  # Features
b = np.array([e[2] for e in equations], dtype=float)           # Targets
```

---

## 📉 Loss Function: Least Squares
We use the **sum of squared errors** as the loss function:

```python
def compute_loss(A, b, z):
    error = A @ z - b
    return np.sum(error**2)
```

---

## 🔍 Gradient Calculation
We manually calculate the gradient of the loss function to update our parameters:

```python
def compute_gradient(A, b, z):
    error = A @ z - b
    return 2 * A.T @ error
```

---

## ⚙️ Gradient Descent Algorithm
We apply gradient descent to iteratively minimize the loss:

```python
learning_rate = 1e-6
num_iterations = 200000
z = np.zeros(A.shape[1])

for i in range(num_iterations):
    gradient = compute_gradient(A, b, z)
    z -= learning_rate * gradient
    if i % 10000 == 0:
        print(f"Iteration {i}, Loss: {compute_loss(A, b, z):.2f}")
```

---

## ✅ Output
After training, the vector `z` contains the best fit parameters that minimize prediction error.

---

## 📚 Concepts Covered
- 📐 Representing linear systems in NumPy
- 🔢 Loss function (Least Squares)
- 📏 Manual gradient computation
- 🔁 Gradient Descent Optimization

---

## 📦 Requirements
- Python 3.x 🐍
- NumPy 🔢

Install dependencies:
```bash
pip install numpy
```

---

## ▶️ How to Run
1. Clone this repo:
```bash
git clone https://github.com/yourusername/linear-regression-from-scratch.git
```
2. Run the script:
```bash
python linear_regression.py
```

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).

---

> 💡 **Tip:** This project is great for learning linear regression internals, understanding how gradient descent works, and seeing how core math applies to real problems.

Made with ❤️ by [Your Name]
