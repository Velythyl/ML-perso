import math
import numpy as np


class GaussianMaxLikelihood:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.mu = np.zeros(n_dims)
        # We only save a scalar standard deviation because our model is the isotropic Gaussian
        # We avons un scalaire comme écart-type car notre modèle est une loi gaussienne isotropique
        self.sigma_sq = 1.0

    # For a training set, the function should compute the ML estimator of the mean and the variance
    # Pour un jeu d'entraînement, la fonction devrait calculer les estimateur ML de l'espérance et de la variance
    def train(self, train_data):
        # Here, you have to find the mean and variance of the train_data data and put it in self.mu and self.sigma_sq
        # Ici, nous devons trouver la moyenne et la variance dans train_data et les définir dans self.mu and self.sigma_sq

        data = train_data[:, :-1]

        self.mu = np.mean(data, axis=0)

        x_moins_mu = np.array([d - self.mu for d in data])
        temp_sigma = 0
        for xu in x_moins_mu:
            temp_sigma += np.dot(np.transpose(xu), xu)
        self.sigma_sq = temp_sigma / (len(data) * self.n_dims)

    def gauss_exp(self, x):
        x_moins_mu = x - self.mu
        norm = np.linalg.norm(x_moins_mu) ** 2
        return (-1 / 2) * norm / self.sigma_sq

    # Returns a vector of size nb. of test ex. containing the log probabilities of each test example under the model.
    # Retourne un vecteur de dimension égale au nombre d'ex. test qui contient les log probabilité de chaque 
    # exemple test
    def loglikelihood(self, test_data):
        # comment the following line once you have completed the calculation of log_prob
        # mettez en commentaire cette ligne lorsque vous avez complétez le calcul de log_prob

        # the following line calculates log(normalization constant)
        # la ligne suivante calcule le log(normalization constant)
        c = - ((self.n_dims / 2) * np.log(2 * np.pi)) - (self.n_dims * np.log(np.sqrt(self.sigma_sq)))
        # It is necessary to calculate the value of the log-probability of each test example
        # under the model determined by mu and sigma_sq. The vector of probabilities is / will be log_prob
        # Il est nécessaire de calculer la log-probabilité de chaque exemple test sous le modèle déterminé
        # par mu et sigma_sq. Le vecteur de probabilité est/sera log_prob

        # ---> WRITE CODE HERE/ÉCRIVEZ VOTRE CODE ICI
        # log_prob =

        g_e = np.array([self.gauss_exp(v) for v in test_data])

        return g_e + c


class BayesClassifier:
    def __init__(self, maximum_likelihood_models, priors):
        self.maximum_likelihood_models = maximum_likelihood_models
        self.priors = priors
        if len(self.maximum_likelihood_models) != len(self.priors):
            print('The number of ML models must be equal to the number of priors!')
        self.n_classes = len(self.maximum_likelihood_models)

    # Returns a matrix of size number of test ex. times number of classes containing the log
    # probabilities of each test example under each model, trained by ML.
    # Retourne une matrice de dimension [nb d'ex. test, nb de classes] contenant les log
    # probabilités de chaque ex. test sous le modèle entrainé par le MV.
    def loglikelihood(self, test_data):

        log_pred = np.zeros((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Here, we will have to use maximum_likelihood_models[i] and priors to fill in
            # each column of log_pred (it's more efficient to do a entire column at a time)
            # Ici, nous devrons utiliser maximum_likelihood_models[i] et priors pour remplir
            # chaque colonne de log_pred (c'est plus efficace de remplir une colonne à la fois) 

            # ---> WRITE CODE HERE/ÉCRIVEZ VOTRE CODE ICI
            log_pred[:, i] = self.maximum_likelihood_models[i].loglikelihood(test_data) * priors[i]

        return log_pred


# In this part, we will train a BayesClassifier on the iris dataset, according to the what we described in the High level description section. We will use the classes BayesClassifier and GaussianMaxLikelihood.
# 
# Contrary to what we did in the previous labs, **we will not use a train/test set**. We will simply use the whole dataset to train the classifier, and we are going to ask the trained classifier to **make predictions on the training set itself**. 
# 
# There are extra questions at the end of this notebook where you can play around and use a proper train/test split. But this will not be tested by gradescope.
# 
# **Complete** the function `get_accuracy(test_inputs, test_labels)`, that takes as input a test set, and returns the trained classifier's (`classifier`) accuracy on this test set. Your function should return a number between 0 and 1.
# 
# To make sure everything works fine, you can test your code by calling your function on the train dataset (This part of the code after the function is provided for you). But remember that your function should work with any inputs, and not only (iris_train, iris_labels).
# 
# <hr>
# 
# Dans cette partie, nous allons entraîner BayesClassifier sur le dataset iris, tel qu'expliqué dans la description détaillée. Nous allons utiliser les classes BayesClassifier et GaussianMaxLikelihood.
# 
# Contrairement à ce que nous avons fait dans le lab précédent, **nous n'allons pas utiliser un dataset train/test**. Nous allons simplement utiliser le dataset complet pour entraîner le modèle et ensuite **faire des prédictions sur le training set lui-même.**
# 
# Des questions bonus se trouvent à la fin de ce notebook pour que vous puissiez explorer avec un ensemble d'entraînement/test. Ces questions ne seront cependant pas testées sur gradescope.
# 
# **Complétez** la fonction `get_accuracy(test_inputs, test_labels)`, qui prend en paramètre un ensemble test, et retourne l'exactitude du classifieur (`classifier`) entraîné. Votre fonction devrait retourner un nombre entre 0 et 1.
# 
# Pour vous assurer que tout fonctionne, vous pouvez tester votre code par appeller votre fonction sur l'ensemble d'entraînement (cette partie du code, après la fonction, vous est donnée). Souvenez-vous cependant que votre fonction devrait fonctionner avec toutes données d'entrée, et pas seulement (iris_train, iris_labels).

# In[ ]:


# We load the dataset and split it into input data, and labels
# On télécharge le dataset et le séparons en input et en étiquettes
import sys
import numpy as np

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    iris = np.loadtxt('http://www.iro.umontreal.ca/~dift3395/files/iris.txt')
else:
    iris = np.loadtxt('iris.txt')
iris_train = iris[:, :-1]
iris_labels = iris[:, -1]

# We split the input data into three sub-datasets, corresponding to the classes 1, 2, and 3. 
# It is necessary to make such a split, in order to train a GaussianMaxLikelihood model per class
# Note that the split (0:50, 50:100, 100:150) is not arbitrary. It corresponds to the three classes in the dataset
# On sépare le input en trois sous-ensembles, correspondant aux classes 1, 2 et 3. Il est nécessaire
# de faire cette partition afin d'entraîner GaussianMaxLikelihood pour chaque classe. Notez que la partition
# (0:50, 50:100 et 100:150) n'est pas arbitraire: elle correspond aux trois classes dans le dataset.
iris_train1 = iris_train[0:50, :]
iris_train2 = iris_train[50:100, :]
iris_train3 = iris_train[100:150, :]

# We create a model per class (using maximum likelihood), and train each of them using the corresponding data
# Nous créons un modèle par classe (en utilisant le maximum de vraisemblance) et les entraînons
model_class1 = GaussianMaxLikelihood(4)
model_class2 = GaussianMaxLikelihood(4)
model_class3 = GaussianMaxLikelihood(4)
model_class1.train(iris_train1)
model_class2.train(iris_train2)
model_class3.train(iris_train3)

# We create a list of all our models, and the list of prior values
# Here the priors are calculated exactly because we know the number of representatives per class.
# Nous créons une liste de nos modèles et une liste de nos priors. Ici, nos priors sont
# déjà calculés car nous savons les proportions par classe
model_ml = [model_class1, model_class2, model_class3]
priors = [1. / 3, 1. / 3, 1. / 3]

# We create our classifier with our list of Gaussian models and our priors
# Nous créons notre classifieur avec notre liste de modèles gaussiens et nos prob à priori
classifier = BayesClassifier(model_ml, priors)


# Returns a number between 0 and 1 representing the accuracy of the model on the test_inputs
# Retourne un nombre entre 0 et 1 représentant l'exactitude du modèle sur les test_inputs
def get_accuracy(test_inputs, test_labels):
    # We can calculate the log-probabilities according to our model
    # Nous pouvons calculez les log-probabilités selon notre modèle

    # ---> WRITE CODE HERE/ÉCRIVEZ VOTRE CODE ICI

    # It now remains to calculate the predicted labels
    # Il reste à calculer les classes prédites

    # ---> WRITE CODE HERE/ECRIVEZ VOTRE CODE ICI

    # Return the accuracy by comparing your predicted labels to the actual labels
    # Retournez l'exactitude en comparant les classes prédites aux vraies étiquettes

    # ---> WRITE CODE HERE/ECRIVEZ VOTRE CODE ICI
    return acc


if __name__ == '__main__':
    print("The training accuracy is : {:.1f} % ".format(100 * get_accuracy(iris_train, iris_labels)))

# ## Once you're done/Une fois terminé (Bonus)

# - Change your code so that `GaussianMaxLikelihood` calculates 1) a diagonal covariance matrix (where we estimate the variance for each component / trait of the input) 2) a full covariance matrix. You can for example have a parameter `cov_type` that can be "isotropic", "diagonal" or "full". The `numpy.cov` and` numpy.var` commands will probably be useful.
# 
# - Instead of using the whole iris dataset to train a `BayesClassifier`, peform a random split of the data (2/3, 1/3) to obtain a train set and a test set, and use the train set to train your classifier. Compare the obtained accuracies on the train set and the test set. Remember to adjust the code that defines `iris_train1`, `iris_train2`, `iris_train3`, and `priors` accordingly. Compare the train and test accuracies of 3 models: one that uses "isotropic" gaussian models, one that uses "diagonal" gaussian models, and one that uses "full" gaussian models.
# 
# - Instead of using the 4 features to train the classifier, use only 2 and try to make nice visualizations as we did in Lab 2. You can of course use the utility functions from there to make your plots.
# 
# <hr>
# 
# - Changez votre code afin que `GaussianMaxLikelihood` calcule 1) une matrice de variance-covariance diagonale (où on estime la variance de chaque élément/trait des données d'entrée) 2) une matrice de variance-covariance complète. Vous pouvez par exemple avoir un paramètre `cov_type` qui peut être `isotropic`, `diagonal` ou `full`. Les commandes `numpy.cov` et `numpy.var` peuvent être utiles.
# 
# - À la place d'utiliser le dataset complet iris pour entraîner `BayesClassifier`, performez une séparation aléatoire de (2/3, 1/3) afin d'obtenir un ensemble d'entraînement et un ensemble test. Comparez les exactitudes du classifieur obtenues à partir des ensembles train et test. Prenez soin d'ajuster le code qui définit `iris_train1`, `iris_train2`, `iris_train3` et `priors` en conséquence. Comparez l'exactitude des ensembles train et test des 3 modèles: gaussienne isotropique, gaussienne diagonale et gaussienne *complète*.
# 
# - Plutôt que d'utiliser les 4 traits afin d'entraîner votre classifieur, utilisez seulement que deux caractéristiques et effectuer des analyses de visualisation telles que faites dans le lab 2. Vous pouvez biensûr faire usages des fonctions du lab 2 pour faire vos graphiques.
