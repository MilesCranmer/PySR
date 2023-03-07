# Tuning and Workflow Tips

I give a short guide below on how I like to tune PySR for my applications.

First, my general tips would be to avoid using redundant operators, like how `pow` and `square` and `cube` are equivalent. The fewer operators the better; only use operators you need.

When running PySR, I usually do the following:

I run from IPython on the head node of a slurm cluster. Passing `cluster_manager="slurm"` will make PySR set up a run over the entire allocation. I set `procs` equal to the total number of cores over my entire allocation.

1. Use the default parameters.
2. Use only the operators I think it needs and no more.
3. Set `niterations` to some very large value, so it just runs for a week until my job finishes. If the equation looks good, I quit the job early.
4. Increase `populations` to `3*num_cores`.
5. Set `ncyclesperiteration` to maybe `5000` or so, until the head node occupation is under `10%`.
6. Set `constraints` and `nested_constraints` as strict as possible. These can help quite a bit with exploration. 
7. Set `maxdepth` as strict as possible.
8. Set `maxsize` a bit larger than the final size you want. e.g., if you want a final equation of size `30`, you might set this to `35`, so that it has a bit of room to explore.
9. Set `parsimony` equal to about the minimum loss you would expect, divided by 5-10. e.g., if you expect the final equation to have a loss of `0.001`, you might set `parsimony=0.0001`.
10. Set `weight_optimize` to some larger value, maybe `0.001`. This is very important if `ncyclesperiteration` is large, so that optimization happens more frequently.
11. Set `turbo` to `True`. This may or not work, if there's an error just turn it off (some operators are not SIMD-capable). If it does work, it should give you a nice 20% speedup.

Since I am running in IPython, I can just hit "q<enter>" to stop the job, tweak the hyperparameters, and then start the search again.

Some things I try out to see if they help:

1. Play around with `complexity_of_operators`. Set operators you dislike (e.g., `pow`) to have a larger complexity.
2. Try setting `adaptive_parsimony_scaling` a bit larger, maybe up to `1000`.
3. Sometimes I try using `warmup_maxsize_by`. This is useful if you find that the search finds a very complex equation very quickly, and then gets stuck. It basically forces it to start at the simpler equations and build up complexity slowly.
4. Play around with different losses:
    i. I typically try `L2DistLoss()` and `L1DistLoss()`. L1 loss is more robust to outliers compared to L2, so is often a good choice for a noisy dataset. 
    ii. I might also provide the `weights` parameter to `fit` if there is some reasonable choice of weighting. For example, maybe I know the signal-to-noise of a particular row of `y` - I would set that SNR equal to the weights. Or, perhaps I do some sort of importance sampling, and weight the rows by importance.

Very rarely I might also try tuning the mutation weights, the crossover probability, or the optimization parameters. I never use `denoise` or `select_k_features` as I find they aren't very useful.

For large datasets I usually just randomly sample ~1000 points or so. In case all the points matter, I might use `batching=True`.

If I find the equations get very complex and I'm not sure if they are numerically precise, I might set `precision=64`.

You might also wish to explore the [discussions](https://github.com/MilesCranmer/PySR/discussions/) page for more tips, and to see if anyone else has had similar questions.