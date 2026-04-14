# TrueSkill

A TrueSkill-style Bayesian skill inference model implemented in C# using [Infer.NET](https://dotnet.github.io/infer/).

## Overview

TrueSkill is a skill rating system developed by Microsoft Research. It models each player's skill as a Gaussian random variable and updates beliefs about skill levels after observing game outcomes. This project demonstrates the core generative model using Infer.NET's probabilistic programming API.

## The Model

The generative story for each game is:

1. Each player has a latent **skill** drawn from a Gaussian prior.
2. In each game, each player draws a **performance** from a Gaussian centred on their skill (same variance used for both).
3. The **outcome** of a game is deterministic: the player with the higher performance wins.

```
Jill's skill  ~ N(120, 25)       Fred's skill  ~ N(100, 1600)
Jill's perf_i ~ N(Jskill, 25)    Fred's perf_i ~ N(Fskill, 25)
outcome_i      = (Jperf_i > Fperf_i)
```

Priors are chosen to reflect prior beliefs:
- Jill: mean 120, variance 25 (relatively confident she is skilled)
- Fred: mean 100, variance 1600 (high uncertainty — could be anywhere)

After conditioning on observed outcomes, Infer.NET runs Expectation Propagation to compute the posterior Gaussian marginals for each player's skill.

### Example output

With one game where **Fred wins** (`outcomes_data = [false]`):

```
Jskill marginal = Gaussian(119.3, 24.73)
Fskill marginal = Gaussian(144.3, 484.8)
```

Fred's posterior mean jumps from 100 to ~144 because his prior was very diffuse and the single win is strong evidence. Jill's mean barely moves because her prior was already tight.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Usage

```bash
# build
dotnet build

# run (Fred wins scenario)
dotnet run --project TrueSkill/TrueSkill.csproj
```

To switch to the **Jill wins** scenario, edit `Program.cs` and toggle the commented line:

```csharp
bool[] outcomes_data = new bool[] { true };  // Jill wins
// bool[] outcomes_data = new bool[] { false }; // Fred wins
```

Multiple games can be modelled by extending the array:

```csharp
bool[] outcomes_data = new bool[] { true, false, true }; // Jill, Fred, Jill
```

## Dependencies

| Package | Version |
|---------|---------|
| `Microsoft.ML.Probabilistic` | 0.4.2504.701 |
| `Microsoft.ML.Probabilistic.Compiler` | 0.4.2504.701 |

## References

- Herbrich, R., Minka, T., & Graepel, T. (2007). [TrueSkill: A Bayesian Skill Rating System](https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html). NeurIPS 2006.
- Bishop, C. M. (2013). [Model-Based Machine Learning](https://www.mbmlbook.com/). (Chapter 2 covers this exact model.)
- [Infer.NET documentation](https://dotnet.github.io/infer/)
