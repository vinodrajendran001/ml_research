Chemical Machine Learning Research
==================================

This is a repository for testing various machine learning methods and applying them to chemical problems.


Install
-------

    $ git clone https://github.com/crcollins/ml_research
    $ cd ml_research
    $ virtualenv env
    $ . env/bin/activate
    $ pip install -r requirements.txt

At this point, it is a matter of collecting the data into the `data/` directory. For the `mol_data` dataset, this involves running the following commands

    $ mkdir data
    $ cd data
    $ git clone https://github.com/crcollins/mol_data

For the `qm7` dataset and the `qm7b` dataset run the following commands.

    $ mkdir data
    $ data
    $ wget http://quantum-machine.org/data/qm7{,b}.pkl


Todo
----

    [ ] Run multiouput with all structures
    [ ] Get back working with neural nets (Caffe?)
    [ ] Improve cross validation
        [ ] Speed (optimize grid search)
            [ ] Random search
            [ ] Gradient Descent type search
        [ ] Remove "bottoming out" of parameters
    [ ] Add bond types to angles/dihedrals/trihedrals
    [ ] forces
    [ ] Restructure code
        [ ] Main running code
        [ ] Split up feature vectors
        [ ] Split up utils functions
    [ ] NN with summation for atoms
    [ ] Add another condition to single_split
    [ ] Parallelize feature vector creation
    [ ]    (Pipeline this?)
    [ ] Encoding of angles

