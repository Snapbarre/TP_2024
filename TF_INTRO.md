# Introduction tensorflow

Pour commencer je recommande d'aller jeter un oeil à la documentation tensorflow : https://www.tensorflow.org/guide/intro_to_graphs?hl=fr#:~:text=TensorFlow%20uses%20graphs%20as%20the,(%22constant%20folding%22).

TensorFlow exploite la notion de graphe pour réaliser des calculs.
* les noeuds du graphe sont les opérations mathématiques ;
* les arêtes du gaphe sont les données multi-dimensionnelles entrantes ou sortantes des noeuds (tensors).

Un tensor a trois propriétés: rang, forme et type
* Rang signifie le nombre de dimensions du Tenseur (un cube ou une case a le rang 3)
* Forme signifie les valeurs de ces dimensions (la boîte peut avoir la forme 1x1x1 ou 2x5x7)
* Type signifie type de données dans chaque coordonnée de Tensor. 

TensorFlow fournit deux niveaux d'API.
* Low level API : TensorFlow core
* High level API : par exemple tf.contrib.learn

Premier example
```
# importation du module tensorflow 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

# creating nodes in computation graph 
node1 = tf.constant(3, dtype=tf.int32) 
node2 = tf.constant(5, dtype=tf.int32) 
node3 = tf.add(node1, node2) 

node4 = tf.constant(3, dtype=tf.float64) 
node5 = tf.constant(5, dtype=tf.float64) 
node6 = tf.multiply(node4, node5) 

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)

# création d'une session tensorflow
sess = tf.Session() 

# evaluation de node3 et node6 et imprime les résultats 
print("la somme de node1 et node2 est :",sess.run(node3))
print("le produit de node4 et node5 est :",sess.run(node6)) 
print("le produit de matrix1 et matrix2 est :",sess.run(product)) 
sess.run(product)

# ferme la session 
sess.close() 

```

si vous ne voulez pas utiliser ```sess.close()``` alors vous pouver
modifier le code de la manière suivante :
```
with tf.Session() as sess:
	print("la somme de node1 et node2 est :",sess.run(node3))
	print("le produit de node4 et node5 est :",sess.run(node6)) 
	print("le produit de matrix1 et matrix2 est :",sess.run(product)) 
```

Remarques :
```tf.constant()``` produit des nodes dont le contenu ne peut être modifier

```tf.variable()``` produit des nodes pouvant contenir des données variables

Illustration de la différence :
```
# importing tensorflow 
import tensorflow as tf 

# creating nodes in computation graph 
node = tf.Variable(tf.zeros([2,2])) 

# running computation graph 
with tf.Session() as sess: 

	# intialisation des variables globales 
	sess.run(tf.global_variables_initializer()) 

	# evaluation des nodes 
	print("Tensor value before addition :\n",sess.run(node)) 

	# modification du tensor par l'addition d'une donnée 
	node = node.assign(node + tf.ones([2,2])) 

	# evaluate node again 
	print("Valeur du tensor après l'addition :\n", sess.run(node)) 
```

La notion de placeholder est très importante. Elle définit et "réserve" une entrée du graphe. Cette entrée sera affectée plus tard lors de la session.
En voici un exemple dans lequel feed_dict est l'option de la méthode qui permet de définir les valeurs des deux placeholders a et b.
```
# importing tensorflow 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

# creating nodes in computation graph 
a = tf.placeholder(tf.int32, shape=(3,1)) 
b = tf.placeholder(tf.int32, shape=(1,3)) 
c = tf.matmul(a,b) 

# running computation graph 
with tf.Session() as sess: 
	print(sess.run(c, feed_dict={a:[[3],[2],[1]], b:[[1,2,3]]})) 

```

## Exemple d'un modèle de régression logistique

Ces quelques lignes de codes permettent de produire un modèle de regression logistique à partir des données iris.
Vous pouvez trouver des informations concernant cette dataset ici : https://scikit-learn.org/stable/datasets/index.html#iris-dataset ou https://www.kaggle.com/uciml/iris

> Question 1 : Analyser précisément cet exemple afin de bien comprendre son fonctionnement. Attacher une importance particulière à la structuration des tableaux de données qui servent à l'apprentissage.

```
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

iris = datasets.load_iris()

# Permet d'obtenir un vecteur vertical des data
Largeur = iris["data"][:, 3:]

# On créé une liste qui contient 1 lorsque la fleur est de type 2 et 0 sinon pour faire une classification
Type = (iris["target"] == 2).astype(np.int)
print("Type : ", Type)

# On va programmer une résolution par descente de gradients 'à la main'
# Paramètre de la descente de gradient
learning_rate = 0.01 
display_step = 1
n_epochs = 10000

# Initialisation des tenseurs de constantes
X = tf.constant(np.c_[np.ones((len(Largeur),1)), Largeur],dtype=tf.float32, name = "X") #X est un tensor qui contient les features 'Largeur' et une colonne de '1' pour le Theta0
y = tf.constant(Type, shape = (len(Largeur),1), dtype = tf.float32, name="y") # y est un tensor qui représente deux classes possibles


# Modèle
# theta est un tensor de 2 variables en colonne initialisées aléatoirement entre -1 et +1
theta = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0),  name = "theta") 

# la prédicton est faite avec la fonction logistique, pred est le tensor de toutes les prédictions
pred = tf.sigmoid(tf.matmul(X,theta)) 

# l'error est le tensor de toutes les erreurs de prédictions
error = pred - y 

# calcule la MSE qui est en fait la valeur que minimise une descente de gradient sur une fonction logistique
mse = tf.reduce_mean(tf.square(error), name="mse") 
nbExemples = len(Largeur)

# Calcul du gradient de l'erreur
gradients = (2/nbExemples) * tf.matmul(tf.transpose(X), error) 

# Definition de la fonction de correction de theta à partir du gradient
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer() # créer un noeud init dans le graphe qui correspond à l'initialisation

# Execution du modèle

with tf.Session() as sess:
    # On Execute le noeud d'initialisation des variables
    sess.run(init) 
    for epoch in range(n_epochs):
    	# affichage tous les 100 pas de calcul
        if epoch % 100 == 0:  
            print("Epoch", epoch, "MSE =", mse.eval())
        # Exécution d'un pas de recalcule de theta avec appels de tous les opérateurs et tensors nécessaires dans le graphe
	sess.run(training_op) 
    best_theta = theta.eval()
print("Best theta : ", best_theta)
```

> Question 2 : Modifier le code afin de produire un modèle de regression logistique sur des données sur la dimension complète du problème.
