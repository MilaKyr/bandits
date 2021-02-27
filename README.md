# Cascading bandits

This is an implementation of the two MAB algorithms from the paper:  [Cascading Bandits for Large-Scale Recommendation Problems by Shi Zong, Hao Ni, Kenny Sung, Nan Rosemary Ke, Zheng Wen, Branislav Kveton](https://arxiv.org/abs/1603.05359)

## Logic
[Picture credits](https://pixabay.com/pl/vectors/kobieta-styl-naukowiec-laboratorium-40987/)
![](https://github.com/milakyr/bandits/blob/main/pics/kobieta_naukowiec_s.png?raw=true)

The main idea is Explorer :D 
This is a class, that basically runs the MAB a few rounds and gathers the results. It also saves them into the experiment folder (don't worry, no need to create such a folder. It will be created in the first run). 
Also, it is fairly customizable, so you can play with it.

### Data
The main data used to develop this library was MovieLens dataset. 
This dataset is also expected to run with default `play.py`.
If you want to change it to something else, by all means, do it. Remember to adjust the data information in `config.py`

If you want to understand the idea of Cascade bandits deeper, please refer to the paper.

## How it works
Make changes to the `congif.py` file, specifying the path to the dataset, bandits, and so on.
Then, you can run the following commands:
1. `make run` - runs the play.py. This is the main script to run two bandit algorithms CascadeLinTS and CascadeLinUBC against each other.
2. `make test` - will run pytest
3. `make test-cov` - test coverage

## Output
The `make run` command will create the folder `experiment/<date&time>`, where you can find:
* cumulative reward for each bandit (plot below)
* general info about the explorer and bandits parameters
* experiment results (basically mean regret for algorithms :D)
![](https://github.com/milakyr/bandits/blob/main/pics/cumulative_rewards_plot.png?raw=true)

This is a project for *fun*, so any input is highly appreciated. 
Also, *if you found something interesting or want to chat about MAB* don't hesitate to reach out!