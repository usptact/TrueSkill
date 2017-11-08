using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;

namespace TrueSkill
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            //bool[] outcomes_data = new bool[] { true }; // Jill wins
            bool[] outcomes_data = new bool[] { false }; // Fred wins

            int numGames = outcomes_data.Length;
            Range n = new Range(numGames);

            // prior beliefs on skill
            var Jskill = Variable.GaussianFromMeanAndVariance(120, 5 * 5);
            var Fskill = Variable.GaussianFromMeanAndVariance(100, 40 * 40);

            // performance for 1+ games for Jill
            var Jperfs = Variable.Array<double>(n);
            Jperfs[n] = Variable.GaussianFromMeanAndVariance(Jskill, 5 * 5).ForEach(n);

            // performance for 1+ games for Fred
            var Fperfs = Variable.Array<double>(n);
            Fperfs[n] = Variable.GaussianFromMeanAndVariance(Fskill, 5 * 5).ForEach(n);

            // game outcomes (true - Jill wins, false - Fred wins)
            var outcomes = Variable.Array<bool>(n);

            // model
            using (Variable.ForEach(n))
                outcomes[n] = Jperfs[n] > Fperfs[n];

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
