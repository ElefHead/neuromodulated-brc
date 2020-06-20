# Neuromodulated Bistable Recurrent Cell

Tensorflow and Keras based implementation of a recurrent cell as well as the corresponding keras RNN layer 
from [**A bio-inspired bistable recurrent cell allows for long-lasting memory**](https://arxiv.org/abs/2006.05252) 
by [Nicolas Vecoven](https://twitter.com/vecovennicolas), 
[Damien Ernst](https://twitter.com/DamienERNST1) and [Guillaume Drion](https://sites.google.com/site/gdrion25/). 



## Usage
To use the library, clone the repo and place `bistablernn` folder inside your project directory.  


The Neuromodulated Bistable Recurrent Cell can be imported as: 
  ```python
  from bistablernn import NBRCell
  ```
  and use the cell to create an RNN layer as per keras API as: 
  ```python
  tf.keras.layers.RNN(NBRCell(num_units))
  ```

Alternatively, a Neuromodulated Bistable _RNN_ layer can be imported and used as: 
  ```python
  from bistablernn import NBR
  ```

  For example: 

  ```python

  model = tf.keras.Sequential([
    NBR(units=num_hidden, input_shape=input_shape),
    tf.keras.layers.Dense(num_classes)
  ])
  ```
---
An [example](https://github.com/ElefHead/neuromodulated-brc/blob/main/notebooks/sequential%20MNIST.ipynb) of training and 
evaluation using MNIST data is in the `notebooks` folder.

## Notes
* The Bistable Recurrent Cell modifies the GRU. My code also inherits keras GRUCell and GRU and overloads 
the functions to reflect the equation changes.
* The implementation is based on my understanding of the equations and modifications. The [authors' implementation can be found here](https://github.com/nvecoven/BRC).

