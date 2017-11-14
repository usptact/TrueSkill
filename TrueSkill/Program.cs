﻿using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;

namespace TrueSkill
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            int[] outcomes_data = new int[] {
                                                1,  // Jill wins
                                                1,  // Jill wins
                                                1,  // Jill wins
                                                1,  // Jill wins
                                                1,  // Jill wins
                                                2,  // Fred wins
                                                2,  // Fred wins
                                                0,  // Draw
                                                0}; // Draw

            int numGames = outcomes_data.Length;
            Range n = new Range(numGames).Named("n");

            // prior beliefs on skill
            var Jskill = Variable.GaussianFromMeanAndVariance(120, 5 * 5).Named("JillSkill");
            var Fskill = Variable.GaussianFromMeanAndVariance(100, 40 * 40).Named("FredSkill");

            // performance for 1+ games for Jill
            var Jperfs = Variable.Array<double>(n).Named("JillPerfs");
            Jperfs[n] = Variable.GaussianFromMeanAndVariance(Jskill, 5 * 5).ForEach(n);

            // performance for 1+ games for Fred
            var Fperfs = Variable.Array<double>(n).Named("FredPerfs");
            Fperfs[n] = Variable.GaussianFromMeanAndVariance(Fskill, 5 * 5).ForEach(n);

            var drawMargin = Variable.GaussianFromMeanAndVariance(1.0, 10.0).Named("drawMargin");
            Variable.ConstrainPositive(drawMargin);

            // game outcomes (0 - Draw; 1 - Jill wins; 2 - Fred wins)
            var outcomes = Variable.Array<int>(n).Named("outcomes");

            // model
            using (Variable.ForEach(n))
            {
                using (Variable.If(Jperfs[n] < Fperfs[n] + drawMargin | Jperfs[n] < Fperfs[n] + drawMargin))
                    outcomes[n] = 0;
                using (Variable.If(Jperfs[n] > Fperfs[n] + drawMargin))
                    outcomes[n] = 1;
                using (Variable.If(Fperfs[n] > Jperfs[n] + drawMargin))
                    outcomes[n] = 2;
            }

            // attaching data
            outcomes.ObservedValue = outcomes_data;

            InferenceEngine engine = new InferenceEngine();

            Gaussian JskillMarginal = engine.Infer<Gaussian>(Jskill);
            Gaussian FskillMarginal = engine.Infer<Gaussian>(Fskill);

            Console.WriteLine("Jskill marginal = {0}", JskillMarginal);
            Console.WriteLine("Fskill marginal = {0}", FskillMarginal);

            Console.WriteLine("Press any key...");

            Console.ReadKey();
        }
    }
}
