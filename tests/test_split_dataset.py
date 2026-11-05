import unittest
import pandas as pd
from data_preprocessing.split_dataset import choose_split, MODES


class SplitTests(unittest.TestCase):
    def test_reproducible_grouped_split_with_class_coverage(self):
        counts=pd.DataFrame([[100+i]*5 for i in range(10)],index=[f'{i:03d}' for i in range(10)],columns=MODES)
        groups=[['000','001']]+[[u] for u in counts.index[2:]]
        first=choose_split(counts,groups,attempts=100)
        self.assertEqual(first,choose_split(counts,groups,attempts=100))
        train,test,_=first
        self.assertFalse(set(train)&set(test))
        self.assertEqual(set(train)|set(test),set(counts.index))
        self.assertEqual('000' in test,'001' in test)
        self.assertTrue((counts.loc[test].sum()>0).all())
        self.assertTrue((counts.loc[train].sum()>0).all())
