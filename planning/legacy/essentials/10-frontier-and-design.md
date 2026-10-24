# 10 · Control the next update

Research asks what to erase, what to write, and how strongly to trust a new observation.

## Can update strength vary by channel?

FG²-GDN ties key and value scaling. Its “+” variant separates the scaling factors.

![FG-squared-GDN uses tied key/value channel scaling; its plus variant separates the factors. The strips depict scaling factors rather than arbitrary rates for every state element.](../figures/visual/10-write-control.svg)

> Finer factors give selective control through a specific update rule.

## Can erasing and writing be controlled separately?

Gated DeltaNet-2 separately gates the erase read and the incoming write target.

![Gated DeltaNet-2 gates the key before the erase read, and gates the incoming value before forming the write target. Their residual corrects the faded state. The gate is not applied after the erase read.](../figures/visual/10-erase-write.svg)

> Related goals can use different algebra; FG²-GDN+ and GDN-2 are distinct rules.

## Should accumulated uncertainty matter?

Kalman Delta Networks track memory uncertainty and observation reliability to influence the correction gain.

![A one-dimensional Kalman illustration: greater prior uncertainty and a reliable new observation can move an estimate farther toward that observation. Internal memory uncertainty is not answer confidence.](../figures/visual/10-uncertainty.svg)

> Internal memory uncertainty is not automatically confidence in the final answer.

## What would count as progress?

Test the behavior you need, then measure the whole system.

![Five different evaluation questions: correction, interference, retention, evidence access and full-system cost. One successful test does not establish the others.](../figures/visual/10-tests.svg)

> Remembering a revised fact, retrieving its evidence and running efficiently require different tests.

These are research proposals with controlled experimental evidence. Use the [mechanism matrix](../appendices/mechanism-matrix.md) to reconnect the whole map.
