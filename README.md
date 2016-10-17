# cbow-ml-autocompletion
Implementation of autocompletion feature using Continuous Bag of Words via Tensorflow

## Overview

This codes implements the "Continuous Bag of Words" model described in this paper: https://arxiv.org/pdf/1411.2738.pdf. The model uses LSTMs to build an RNN that can take a context of words and predict the most likely one to autocomplete. In this implementation instead of words we use bigrams. Even with low training the model does quite well in making real words and even catch word-to-word dependencies.

The LSTMs are customly built for practice purposes. You can use the ones built-in in Tensorflow for better performance/easy construction of deeper DNN.

## Possible tunings

Currently we use a relatively shallow NN (only two layers) and 10 unrollings. Adding more layers might be interesting, as well as introducing dropouts to allowe more epochs.

## Results

As you can see from the output the network is already creating meaningful words just by the context of the last N bi-grams. You can also see some interword dependencies which mostly visible between numbers, e.g. `own two zero four zero` and common pattern of words: `of the`, 

```
Average loss at step 24000 : 3.25057203054 learning rate: 0.1
Minibatch perplexity: 23.28
================================================================================
pilling relation nish early prevacence of tyunigin creecomators nekur japaevo ofterpor on all cartary toy condexter million spide of the pautain of ban project 
has that crance known stand righon in silvers for the four five nine seef commermed adder chas own two zero four zero and donamiup or seven ital braw the refeun
shitum saintzanca howeven bc starles iggory of the often scientions write of the but to the animental kudy for arglanist day one six one five eight three one ni
zbwarary age last of its uped quaker of its telts keuanciment estication to remoughwal and never the citient cell b one nine five seven two zero zero zero one n
wetical general usion s match of metra ritormon versedleg in the cribon of the first know s modern power of the speating which commusing rudate itage shifisul o
================================================================================
Validation set perplexity: 17.11
```

Results from a similar model but used to predict a missing word in the middle of 4 words (not part of the code, but can be opensourced on requests):
```
Nearest to eight: cheka, gurps, chip, fortran, function, production, desu, satan,
Nearest to time: period, times, speeds, least, night, controversy, distances, distance,
Nearest to his: her, their, its, my, our, your, s, the,
Nearest to its: their, his, her, the, our, taos, your, whose,
Nearest to if: when, though, where, unless, therefore, then, before, since,
Nearest to up: off, out, easier, lobes, down, back, blending, together,
Nearest to world: falklands, moot, fiorello, nation, censured, u, stills, narrator,
Nearest to while: although, when, though, but, however, where, and, after,
Nearest to who: whom, he, tsinghua, whose, young, which, gifford, irritate,
Nearest to no: little, only, any, confessor, prophylactic, always, still, psychoanalytic,
Nearest to use: form, study, usage, play, apply, contribution, village, addition,
Nearest to to: towards, will, should, must, could, hemispheric, would, can,
Nearest to one: terms, novices, wizardry, hongkong, imitates, chico, hawthorne, epp,
Nearest to between: with, among, within, through, across, infamously, beyond, mateo,
Nearest to which: that, where, this, what, usually, also, typically, when,
Nearest to more: less, rather, larger, smaller, fewer, very, worse, better,

"anarchism originated X a term" most likely fit: ['in', 'as', 'for', 'and', 'to', 'that', 'with', 'of']
```
As you can see the model starts associate that there's a correlation between `time` and `distance`, also `as` is the second best prediction for the missing word in the middle of the `"anarchism originated X a term` sentence.

## Requirements
    python 2.7
    tensorflow 0.6.0

Tensorflow can be installed using

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
    pip install --ignore-installed --upgrade $TF_BINARY_URL
