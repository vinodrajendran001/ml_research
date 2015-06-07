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
    [ ] NN with summation for atoms
