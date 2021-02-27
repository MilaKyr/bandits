# Cascading bandits

This is an implementation of the two MAB algorithms from the paper:  [Cascading Bandits for Large-Scale Recommendation Problems by Shi Zong, Hao Ni, Kenny Sung, Nan Rosemary Ke, Zheng Wen, Branislav Kveton](https://arxiv.org/abs/1603.05359)

## Logic with code snippets
[Picture credits](https://pixabay.com/pl/vectors/kobieta-styl-naukowiec-laboratorium-40987/)
![](https://github.com/milakyr/bandits/blob/main/pics/kobieta_naukowiec.png?raw=true)
The main idea is Explorer :D It is fairly customizable, so you can play with it.

### Data
The main data used to develop this library was MovieLens dataset. 
This dataset is also expected to run with default `play.py`.
If you want to change it to something else, by all means do it. Remember to adjust the data information in `config.py`

If you want to understand more deeply the idea od Cascade bandits, please refer to the paper.

## How it works
Fist of all, make changes to the `congif.py` file, specifying path to dataset, bandits and so on.
Then, you can run the following commands:
1. `make run` - runs tha play.py. This is the main script to run two bandit algorithms CascadeLinTS and CascadeLinUBC against each other.
2. `make test` - will run pytest
3. `make test-cov` - test coverage

This is a project fo *fun*, so any input is highly appreciated. Also, *if you found somethind interesting or want to chat about MAB* don't hesitate to reach out!
