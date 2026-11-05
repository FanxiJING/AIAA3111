import unittest
import pandas as pd
from data_preprocessing.filter_modes import filter_points
from feature_enginneering.motion_features import build_features, FEATURES


class FilterTests(unittest.TestCase):
    def test_removed_classes_break_features_but_target_changes_do_not(self):
        d = pd.DataFrame({'user_id':['001']*7, 'trip_id':['a']*6+['b'],
                          'mode':['walk','bus','subway','walk','airplane','car','train'],
                          'timestamp':pd.to_datetime([0,1,2,3,4,5,2000],unit='s'),
                          'latitude':[0.]*7, 'longitude':[0,.001,.002,.003,.004,.005,.006]})
        f=filter_points(d)
        self.assertEqual(f['mode'].tolist(),['walk','bus','walk','car','train'])
        self.assertEqual(f.feature_block_start.tolist(),[True,False,True,True,True])
        self.assertEqual(f.trip_id.tolist(),['a','a','a','a','b'])
        r=build_features(f)
        self.assertTrue(r.loc[r.feature_block_start,FEATURES].isna().all().all())
        self.assertTrue(r.speed_mps_valid.iloc[1])
        self.assertEqual(len(filter_points(d.iloc[:0])),0)
