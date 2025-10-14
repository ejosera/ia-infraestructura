# Python for Data Science and Machine Learning Bootcamp – Jose Portilla (Udemy)
##  Introduction to python crash couse

### printing
```
num = 12
name = 'Sam'

print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))
My number is: 12, and my name is: Sam

print('My number is: {}, and my name is: {}'.format(num,name))
My number is: 12, and my name is: Sam
```


### List: Ordered, Mutable Collection
- **Ordered:** Elements maintain their insertion order.  
- **Mutable:** You can modify, add, or remove elements.  
- **Allows duplicates.**


```python
my_list = [1, 2, 3, 3, "hello"]
print(my_list[0])  # 1

my_list.append(4)
print(my_list)     # [1, 2, 3, 3, 'hello', 4]
```

### Tuple: Ordered, Immutable Collection
- **Ordered:** Like lists, order is preserved
- ** Immutable:** Once created, it cannot be changed
- ** Allows duplicates**
```
my_tuple = (1, 2, 3)
print(my_tuple[1])  # 2

# my_tuple[1] = 5  # ❌ Error – tuples can’t be modified
```

### Dictionary (dict): Key–Value Pairs
- **Unordered** (since Python 3.7 → insertion-ordered)
- **Mutable**: You can add, remove, or change key-value pairs
- **Keys must be unique and immutable**

```
my_dict = {"name": "Alice", "age": 30, "city": "Paris"}
print(my_dict["name"])  # Alice

my_dict["age"] = 31     # Update
my_dict["country"] = "France"  # Add new key
print(my_dict)
# {'name': 'Alice', 'age': 31, 'city': 'Paris', 'country': 'France'}
```

### Set: Unordered Collection of Unique Items
- **Unordered:** No index or defined order
- **Mutable:** You can add/remove items
- **No duplicates allowed**
```
my_set = {1, 2, 2, 3, 4}
print(my_set)  # {1, 2, 3, 4}

my_set.add(5)
my_set.remove(3)
print(my_set)  # {1, 2, 4, 5}
```

### Functions
Code used more than once o for clarity
```
def my_func(var)
  """
  THIS IS A DOC STRING
  Info providede when Shift + Tab
  """
  print('Hello ' +name)

my_func(José)
Hello José


```

# Python for Data Analysis  - Numpy
## Introducción
Numpy trabaja con Numpy Arrais
- Vectors (1 d array)
- Array (matriz)

## Numpy Array
### Creo un numpy array
```
my_list = [1,2,3]
np.array(my_list)

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
np.array(my_matrix)

np.arange(0,10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(0,11,2)
array([ 0,  2,  4,  6,  8, 10])

np.zeros(3)
array([ 0.,  0.,  0.])

np.zeros((5,5))
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])

np.ones((3,4))
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
```

### linspace Creo un array con numberos separados de forma equidistante
```
np.linspace(0,10,3)
array([ 0.,  5., 10.])
```

### Matriz identidad
```
np.eye(4)
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])
```

### Random
rand crea números entre 0 y 1
```
np.random.rand(5,5)
array([[0.44929898, 0.89160936, 0.40444023, 0.02757594, 0.08884555],
       [0.47367313, 0.11225391, 0.4955617 , 0.28784571, 0.66607695],
       [0.47367866, 0.5194453 , 0.35563614, 0.64282024, 0.04975028],
       [0.06815864, 0.23308848, 0.18330004, 0.65306233, 0.76903471],
       [0.99048808, 0.69580378, 0.89698572, 0.12230147, 0.9922881 ]])
```

randn crea números según una distribución normal centrada en 0
```
np.random.randn(5,5)
array([[ 0.13472148, -0.75450279, -0.97847649,  0.66480845, -1.05863038],
       [ 1.48196232,  0.99722198, -0.62302153, -0.51158007,  0.29080428],
       [ 1.23855807,  0.60210283,  0.94860448,  0.82414583, -0.61187368],
       [-0.79993548,  1.19655447, -0.55748516, -1.39554422, -0.58397525],
       [-0.20119201, -0.2844824 ,  0.06018214, -1.33721035, -1.75851488]])
```

randint crea números enteros desde low hasta pero sin incluir high
```
np.random.randint(1,100,10)
array([94, 79, 82, 41, 99, 35, 46, 68, 74, 15])
```

### Atributos y métodos
Reshape convierte un vector 1,m en una matriz cuadrada siempre que nxn=m
```
arr = np.arange(25)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
arr.reshape(5,5)


array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
```

Obtener el min, max y sus indices
```
ranarr = np.random.randint(0,50,10)
array([10, 12, 41, 17, 49,  2, 46,  3, 19, 39])

ranarr.max()
49
ranarr.argmax()
4
ranarr.min()
2
ranarr.argmin()
5
```

Shape dice el tamaño de la matriz y reshape cambia el tamaño
```
arr.shape
(25,)

arr.reshape(25,1)


array([[ 0],
       [ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10],
       [11],
       [12],
       [13],
       [14],
       [15],
       [16],
       [17],
       [18],
       [19],
       [20],
       [21],
       [22],
       [23],
       [24]])
```

### Tipo de array
```
arr.dtype
dtype('int64')
```

## Numpy array index
creo un arrai
```
arr = np.arange(0,11)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

### Indexing
Obtengo un rango de valores
```
arr[1:5]

array([0, 1, 2, 3, 4])
```

### Broadcasting
Asigno un valor a un rango de un array
```
arr[0:5]=100
array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])
```

**Slice** de un array no es una copia es una imagen del array original
```
arr = np.arange(0,11)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

slice_of_arr = arr[0:6]
array([0, 1, 2, 3, 4, 5])

slice_of_arr[:]=99
array([99, 99, 99, 99, 99, 99])

arr
array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])
```

Si quiero copiar un array lo tengo que decir explicitamente
```
arr_copy = arr.copy()
```

### indexing a 2D array (matriz)
```
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
array([[ 5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])

arr_2d[1]
array([20, 25, 30])
```

Hay 2 formatos
- arr_2d[row][col]
- arr_2d[row,col]

```
arr_2d[1][0]
20

arr_2d[1,0]
20

arr_2d[:2,1:]
array([[10, 15],
       [25, 30]])
```

### Seleccionar una matriz
```
arr2d
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
       [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
       [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
       [6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
       [7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
       [8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
       [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]])

arr2d[[6,4,2,7]]
rray([[ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
       [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
       [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
       [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.]])
```

### Selección por comparación
```
arr = np.arange(1,11)
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr > 4
array([False, False, False, False,  True,  True,  True,  True,  True,  True], dtype=bool)

bool_arr = arr>4
array([False, False, False, False,  True,  True,  True,  True,  True,  True], dtype=bool)

arr[bool_arr]
array([ 5,  6,  7,  8,  9, 10])
```

o en un paso
```
arr[arr>2]
array([ 3,  4,  5,  6,  7,  8,  9, 10])
```

## Operaciones en numpy
```
arr = np.arange(0,10)

arr + arr
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
```

** numpy da un warning no un error al dividir por cero o da infinito si 1/0
```
arr/arr
/tmp/ipython-input-2878212635.py:3: RuntimeWarning: invalid value encountered in divide
  arr/arr

array([nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
```

[Funciones universales](https://www.google.com/url?q=http%3A%2F%2Fdocs.scipy.org%2Fdoc%2Fnumpy%2Freference%2Fufuncs.html
)

```
np.sqrt(arr)
np.exp(arr)
np.max(arr) #same as arr.max()
np.sin(arr)
np.sin(arr)
```

