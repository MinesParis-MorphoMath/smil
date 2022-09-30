1. style de code
   * emacs whitespace-mode
   * clang-format/git-clang-format
	 + adapter la position des commentaires
2. éditeur++
   * coloration syntaxique
   * vérification syntaxique -> détection des warnings
   * auto-complémentation
   * refactoring
   * goto definition
   * preprocessing
   * fixit
   * diff-buffers
3. gestion des sources
   * éviter de garder/commiter du code mort
   * .gitignore
   * privilégier une seule branche si pas à l'aise
   * commit régulièrement (~5 fois par jour)
   * ne pas garder dand un état sale (avec des fichiers en instance de commit)
	 + merge conflicts
4. organisation du dépôt
   * un sous-niveau supplémentaire par rapport aux autres modules
6. tests
   * indépendants de la machine (pas de chemins en dur)
   * augmenter la couverture
   * vérifier/corriger les tests pour conserver leur pertinence
7. structure du code
   * pas de copier-coller ! éviter les blocs de code identiques
   * éviter le code mort et le code commenté
   * vérifier les includes
   * utiliser les concepts de programmation orientée objet
	 + diviser par 2 la taille du code
	 + composition plutôt qu'héritage
	 + supprimer DendroNode.hpp et Dendrogram.hpp ?
   * ajouter plus de documentation format doxygen
   * switch/case sur des enum plutôt que sur des strings
