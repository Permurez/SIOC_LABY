import numpy as np


Sx= np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Sy= np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
L=np.array([[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]])     
G=1/16 * np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])
W=np.array([[ 0, -1, 0],
            [ -1,  5, -1],
            [ 0, -1, 0]])
#rozmycie za pomocą G 
#wykrywanie krawędzi za pomocą używania splotu 
#wyostrzanie to podkręcanie krawędzi z rozmywciem tła