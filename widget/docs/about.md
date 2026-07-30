# Virtual nitrate sensors for Iowa

Iowa's rivers carry nitrate from fertiliser runoff. Above **10 mg/L** the water is deemed unsafe to drink, and the sensor network that measures it is sparse, expensive, and shrinking under budget cuts — most of the state is simply unmonitored.

This tool aims to answer the question a missing sensor would: **on a given day, at a place with no instrument, how much nitrate is in the water and is it over the limit?** It uses only public data — weather, satellite land cover, and soil nitrogen surplus — so it works anywhere in the state.

This was completed at the Summer 2026 Erdos Institute Data Science Bootcamp. Team members [Isaac Martin](https://web.ma.utexas.edu/users/ikmartin/), Rajpreet Kaur and Erin Bevilacqua. Ongoing work occuring in a separate repo, this tool remains a static representation of our work and a proof-of-concept.

## Try it

1. Choose **Pin drop** in the Selection panel, then click anywhere on an Iowa waterway.
2. Pick a year and press **Run forecast**.
3. Drag the **β slider** to trade catching more violations against raising more false alarms.

The pin snaps to the nearest mapped stream reach, and a dashed line shows where it moved from. The forecast describes that reach's drainage basin, not the exact point you clicked.

## How well does it work?

Evaluated on **basins the model never saw in training**, the honest test for a location with no sensor:

| | |
|---|---|
| Ranks violation days | **2.7× better than chance** (average precision 0.69, against a 26% base rate) |
| Catches violations | **88%** of them, with 53% of alarms false, at the default β = 2 |
| Predicts concentration | **R² 0.37**, typical error 4.4 mg/L |

Trained on 81 sensors and 158,215 sensor-days, validated across 20 hydrologically independent basin families.

## What it is not

It is not a substitute for a physical sensor. It is weakest at ranking one basin's *overall* level against another's, and strongest at timing *when* nitrate rises within a basin. Where the realistic alternative is no information at all, that is still a useful trade.

It also assumes stringent data restrictions: it does not make use of the data provided by physical sensors in the water at prediction time. Ongoing work aims to build on this proof-of-concept to produce a robust nationwide virtual nitrate model intended to supplement an existing in-situ nitrate sensor network.

## Under the hood

Everything you are looking at runs in your browser. All 16,760 stream reaches were delineated ahead of time and their features precomputed, the gradient-boosted trees are repacked into a binary blob and walked in JavaScript, and the page itself is a static snapshot. The model is light enough that this barely affects performance.

Ongoing work is occuring at this [forked repo](https://github.com/ikmartin/sustag), and we aim to deploy those (heaftier) models on a dedicated server.

Read the [executive summary](executive_summary.md) for the full account, [kpis.md](kpis.md) for how the scores are defined, or the [source on GitHub](https://github.com/Erdos-Projects/summer26-sustainability-of-agriculture).
