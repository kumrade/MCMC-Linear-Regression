MCMC Linear Regression


MCMC stands for Markov Chain Monte Carlo. Fancy name, simple idea:

Imagine wandering randomly through a space of possible answers (lines) — but spending more time where the answers make more sense.

You do this "wandering" over and over until you've gathered a good picture of where the best answers are hiding.

💥 Why do we need it?
In Bayesian regression, we want to know what the posterior distribution (our belief after seeing the data) looks like.

But sometimes, the math to calculate it directly is too messy or impossible. So instead of calculating it directly, we simulate it with MCMC — like rolling dice many times to approximate the outcome probabilities.

🧪 The Real-Life Analogy
Think of it like this:

You're blindfolded and dropped into a hilly field (each point in the field is a possible intercept & slope).

Higher ground = better fit to the data (more likely solution).

You take a random step (try a new pair of values).

If the new spot is higher, you go there.

If it’s lower, you might still go there — but less likely.

You keep doing this again and again.

After a while, you spend most time on the high grounds — places that give a better fit.

That's Metropolis-Hastings, the MCMC algorithm used here.

🔍 What the code does (in plain English):
Start with a random guess for the intercept and slope (say 0.5 and 0.5).

At each step:

Make a small random tweak to the guess.

Calculate how likely this new guess explains the data (posterior probability).

Decide whether to accept the new guess based on how much better (or worse) it is.

Repeat 50,000 times.

Throw away the first 10,000 steps (because the model is still “warming up”).

The rest give a good picture of where the true parameters likely lie.

📈 What do we get in the end?
You get a histogram of guesses for both:

Intercept

Slope

These histograms show where the "best guesses" live. You also calculate the average of all those good guesses to get your final regression line.

✅ Result:
MCMC gives you intercept ≈ 0.50

Slope ≈ 2.0

Pretty close to the actual values we used to generate the data!
