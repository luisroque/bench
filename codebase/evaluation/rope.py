import numpy as np
import pandas as pd


class RopeAnalysis:

    @staticmethod
    def get_vector_probs(diff_vec: pd.Series, rope: float):
        left = (diff_vec < -rope).mean()
        right = (diff_vec > rope).mean()
        mid = np.mean([-rope < x_ < rope for x_ in diff_vec])

        return left, mid, right

    @classmethod
    def get_probs(cls,
                  results: pd.DataFrame,
                  rope: float,
                  reference: str):
        results_pd = cls.calc_percentage_diff(results, reference)

        prob_df = results_pd.apply(lambda x: cls.get_vector_probs(x, rope=rope), axis=0)
        prob_df = prob_df.T.reset_index()

        outcome_names = [f'{reference} loses', 'draw', f'{reference} wins']

        prob_df.columns = ['Method'] + outcome_names

        loc = prob_df.query(f'Method=="{reference}"').index[0]

        prob_df = prob_df.drop(loc).reset_index(drop=True)

        df_melted = prob_df.melt('Method')
        df_melted['variable'] = pd.Categorical(df_melted['variable'], categories=outcome_names)

        df_melted.columns = ['Model', 'Result', 'Probability']

        return df_melted

    @classmethod
    def calc_percentage_diff(cls, results: pd.DataFrame, reference: str):
        results_pd = results.copy()
        for c in results:
            results_pd[c] = cls.percentage_diff(results[c],
                                                results[reference])

        return results_pd

    @staticmethod
    def percentage_diff(x, y):
        return ((x - y) / abs(y)) * 100
