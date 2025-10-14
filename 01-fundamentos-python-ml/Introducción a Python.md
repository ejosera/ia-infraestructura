# 🐍 Guía Práctica de Fundamentos de Python

Una introducción completa y sencilla a las características más importantes de **Python**, con ejemplos y ejercicios para practicar.

---

## 🖨️ `print()` – Mostrar información

```python
print("Hola, mundo!")
nombre = "Ana"
print("Hola,", nombre)
```

📤 **Salida:**

```
Hola, mundo!
Hola, Ana
```

### 🏋️ Ejercicio

Imprime tu nombre, edad y ciudad en una sola línea usando `print()`.

---

## 📋 Listas (`list`)

Colecciones **ordenadas y mutables**.

```python
frutas = ["manzana", "banana", "cereza"]
frutas.append("naranja")
print(frutas[0])       # 'manzana'
print(frutas[-1])      # 'naranja'
print(len(frutas))     # 4
```

📤 **Salida:**

```
manzana
naranja
4
```

### 🏋️ Ejercicio

1. Crea una lista con tres colores.
2. Agrega un color nuevo.
3. Imprime el segundo elemento.

---

## 🔒 Tuplas (`tuple`)

Colecciones **ordenadas e inmutables**.

```python
coordenadas = (10, 20)
print(coordenadas[0])  # 10
```

### 🏋️ Ejercicio

Crea una tupla con tres números y muestra el último valor.

---

## 🧱 Diccionarios (`dict`)

Estructuras **clave → valor**.

```python
persona = {"nombre": "Juan", "edad": 30, "ciudad": "Madrid"}
print(persona["nombre"])
persona["edad"] = 31
```

📤 **Salida:**

```
Juan
```

### 🏋️ Ejercicio

Crea un diccionario con tus datos (nombre, edad, país) y cambia uno de los valores.

---

## 🧺 Conjuntos (`set`)

Colecciones **no ordenadas** y **sin duplicados**.

```python
numeros = {1, 2, 2, 3, 4}
print(numeros)  # {1, 2, 3, 4}
```

### 🏋️ Ejercicio

Crea un `set` con algunos números repetidos y observa el resultado.

---

## 🧠 Condicionales (`if`, `elif`, `else`)

```python
edad = 20
if edad >= 18:
    print("Eres mayor de edad")
elif edad == 17:
    print("Casi mayor de edad")
else:
    print("Menor de edad")
```

📤 **Salida:**

```
Eres mayor de edad
```

### 🏋️ Ejercicio

Pide una edad por teclado y di si la persona es mayor o menor de edad.

---

## 🔁 Bucles (`for`, `while`)

### 🔹 `for`

```python
frutas = ["manzana", "banana", "cereza"]
for fruta in frutas:
    print(fruta)
```

📤 **Salida:**

```
manzana
banana
cereza
```

### 🔹 `while`

```python
i = 1
while i <= 3:
    print(i)
    i += 1
```

📤 **Salida:**

```
1
2
3
```

### 🏋️ Ejercicio

Usa un bucle `for` para imprimir los números del 1 al 10.

---

## ⚙️ Funciones (`def`)

```python
def saludar(nombre):
    return f"Hola, {nombre}!"

print(saludar("María"))
```

📤 **Salida:**

```
Hola, María!
```

### 🏋️ Ejercicio

Crea una función que reciba dos números y devuelva su suma.

---

## ⚡ Funciones Lambda

Funciones **anónimas y cortas**.

```python
cuadrado = lambda x: x ** 2
print(cuadrado(5))
```

📤 **Salida:**

```
25
```

### 🏋️ Ejercicio

Crea una lambda que devuelva el doble de un número.

---

## 🧮 Comprehensions

```python
numeros = [1, 2, 3, 4, 5]
cuadrados = [x**2 for x in numeros]
print(cuadrados)
```

📤 **Salida:**

```
[1, 4, 9, 16, 25]
```

### 🏋️ Ejercicio

Crea una lista con los cubos de los números del 1 al 5.

---

## 🧾 Manejo de cadenas (`str`)

```python
texto = "Python es genial"
print(texto.upper())     # 'PYTHON ES GENIAL'
print(texto.lower())     # 'python es genial'
print(texto.split())     # ['Python', 'es', 'genial']
print("genial" in texto) # True
```

### 🏋️ Ejercicio

Toma una frase e imprime cuántas palabras tiene usando `.split()`.

---

## 📦 Importar módulos

```python
import math
print(math.sqrt(16))  # 4.0
```

### 🏋️ Ejercicio

Usa el módulo `math` para calcular el seno de 45 grados.

---

## 🧰 Tipos de datos comunes

| Tipo    | Descripción | Ejemplo          |
| ------- | ----------- | ---------------- |
| `int`   | Entero      | `10`             |
| `float` | Decimal     | `3.14`           |
| `str`   | Texto       | `'hola'`         |
| `bool`  | Booleano    | `True` / `False` |
| `list`  | Lista       | `[1, 2, 3]`      |
| `tuple` | Tupla       | `(1, 2, 3)`      |
| `dict`  | Diccionario | `{'a': 1}`       |
| `set`   | Conjunto    | `{1, 2, 3}`      |

---

## 🧠 Características Clave de Python

✅ Sintaxis simple y legible
✅ Tipado dinámico
✅ Gran ecosistema de librerías (NumPy, Pandas, TensorFlow...)
✅ Soporte para OOP y programación funcional
✅ Ideal para automatización, IA, análisis de datos y desarrollo web

---

## 💡 Consejo final

Usa `help(objeto)` o `dir(objeto)` para explorar las funciones disponibles:

```python
help(str)
dir(list)
```

---

📘 **Autor:** Tu asistente de IA 🤖
📅 **Versión:** 1.0 – Guía práctica Python para principiantes.

