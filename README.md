# Dynamic_Local_Regret_for_Non-convex_Online_Forecasting_NeurIPS2019

This repository contains implementation details and the preprint of our paper **Dynamic Local Regret for Non-convex Online Forecasting** by Sergul Aydore, Tianhao Zhu and Dean Foster to appear at NeurIPS 2019.

## Abstract:

We consider online forecasting problems for non-convex machine learning models. Forecasting introduces several challenges such as (i) frequent updates are necessary to deal with concept drift issues since the dynamics of the environment change over time, and (ii) the state of the art models are non-convex models. We address these challenges with a novel regret framework. Standard regret measures commonly do not consider both dynamic environment and non-convex models. We introduce a local regret for non-convex models in a dynamic environment. We present an update rule incurring a cost, according to our proposed local regret, which is sublinear in time $T$. Our update uses time-smoothed gradients. Using a real-world dataset we show that our time-smoothed approach yields several benefits when compared with state-of-the-art competitors: results are more stable against new data; training is more robust to hyperparameter selection; and our  approach is more computationally efficient than the alternatives.

## Implementation Details:

Simply run `Demo.ipynb` in `./code`. Feel free to contact us for any questions or comments.