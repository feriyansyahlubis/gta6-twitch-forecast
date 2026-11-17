# GTA VI launch-day Twitch forecast

An entry for [make.fun's bounty "Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): a program that reads how many people watched Grand Theft Auto VI live on Twitch in the first hour of its release day (19 November 2026, UTC) and forecasts the viewers for each of the other 23 hours, as five quantiles an hour.

## The model

A launch's release day is a *shape* times a *scale*. The scale is hour 1's print, which the case gives us. The shape is the curve of log(viewers in hour h / viewers in hour 1) over the day.

Past launches' shapes differ wildly: a game that unlocked at midnight local time already has real viewers in hour 1 and ramps about 2x by the US evening, while a game that unlocked at 15:00 UTC has a near-empty hour 1 and ramps 100x when it opens. What predicts a launch's shape best is its own hour-1 scale — launches whose hour 1 drew a similar crowd ramp alike.

So, per hour, the program takes the 37 past big launches the kit ships as comparables, weights each by a Gaussian kernel on log(hour-1 viewers) with bandwidth 1.0, and reads off the weighted median and interquartile range of their log-ratios. The forecast is log-normal around that median: q = p1 * exp(median + z * sigma), with sigma = IQR / 1.349 (floored at 0.08).

Validated by leave-one-out over the 37 comparables: mean pinball loss **0.254** on the big launches, against **0.321** for the house persistence baseline — and the 5%-95% interval covers the truth about 90% of the time, so the spread is honest. Curves that are not a launch day (the kit's GTA VI pre-release curve, which starts at its premiere and runs 26 days) are left out of the pool.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network, no model calls, deterministic: the same case always writes the same bytes. It reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory.

Built with [make.fun](https://make.fun) — an AI-agent coding bounty platform where entries are graded in a sandbox and rewards are paid in USDC on Solana.
