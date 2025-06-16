# Multi Taxi Environment

<p>
    <img src="images/taxi_env.png" width="500" alt="taxi env example map"/>
</p>

`multi-taxi` is a highly configurable multi-agent environment, based on [gym](https://www.gymlibrary.ml/)'s
[taxi environment](https://www.gymlibrary.ml/environments/toy_text/taxi/), that adheres to the
[PettingZoo](https://www.pettingzoo.ml/) API. Some configurations include:
1. the number of taxis and passengers in the environment (limited to the size of the map)
2. the domain map itself
3. the environment objective
4. individual taxi configurations:
   1. reward function
   2. action and observation space
   3. passenger and fuel capacity
5. and so much more!

For a quickstart guide and a deeper dive into the environment and its configuraions, please consult our
[demonstration notebook](https://github.com/CLAIR-LAB-TECHNION/multi-taxi/blob/main/notebooks/MultiTaxiEnvDemo.ipynb), also
available in
[colab](https://colab.research.google.com/github/sarah-keren/multi-taxi/blob/main/notebooks/MultiTaxiEnvDemo.ipynb) and
[nbviewer](https://nbviewer.org/github/sarah-keren/multi-taxi/blob/main/notebooks/MultiTaxiEnvDemo.ipynb).
 
# Installation
The easiest way to install `multi-taxi` is directly from the git repository using `pip`. Here is how to install the
latest stable version:
```shell
pip install "git+https://github.com/CLAIR-LAB-TECHNION/multi-taxi@0.5.0"
```

You can also download our latest updates by not specifying a tag, like so:
```shell
pip install "git+https://github.com/CLAIR-LAB-TECHNION/multi-taxi"
```

If you wish to install the environment that uses the legacy pettingzoo API, please install version `0.3.0` like so:
```shell
pip install "git+https://github.com/CLAIR-LAB-TECHNION/multi-taxi@0.3.0"
```

If you are seeking the legacy version, which is based on the [RLLib](https://docs.ray.io/en/latest/rllib/index.html)
API, please install version `0.0.0` like so:
```bash
pip install "git+https://github.com/CLAIR-LAB-TECHNION/multi-taxi@0.0.0"
```

# Acknowledgements
This library is based on [MultiTaxiLib](https://github.com/ofirAbu/MultiTaxiLib) by Ofir Abu. The original
implementation paper can be found [here](https://github.com/ofirAbu/MultiTaxiLib/blob/master/MultiTaxiLabProject.pdf). 

# Citation
To cite this repository in academic works or any other purpose, please use the following BibTeX citation:
```BibTeX
@article{azranContextualPreplanningReward2024,
  title = {Contextual {{Pre-planning}} on {{Reward Machine Abstractions}} for {{Enhanced Transfer}} in {{Deep Reinforcement Learning}}},
  author = {Azran, Guy and Danesh, Mohamad H. and Albrecht, Stefano V. and Keren, Sarah},
  year = {2024},
  month = mar,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {38},
  number = {10},
  pages = {10953--10961},
  issn = {2374-3468},
  doi = {10.1609/aaai.v38i10.28970},
}
```
Alternatively, we offer a [CITATION.cff file](https://citation-file-format.github.io/) with GitHub and Zotero
integration.
