import unittest
import pandas as pd
from feature_enginneering.motion_features import build_features, FEATURES
from feature_enginneering.quality_features import repair_connections


class QualityTests(unittest.TestCase):
    def test_breaks_and_derivative_recovery(self):
        d=pd.DataFrame({'user_id':['001']*8,'trip_id':['a']*8,'feature_block_id':['b']*8,
                        'feature_block_start':[True]+[False]*7,'mode':['walk']*8,
                        'source_file':['x']*6+['y']*2,
                        'timestamp':pd.to_datetime(list(range(8)),unit='s'),
                        'latitude':[0.]*8,'longitude':[0,.00001,.1,.10001,.10002,.10003,.10004,.10005]})
        before=build_features(d)
        after=repair_connections(before)
        self.assertEqual(len(after),8)
        self.assertEqual(after.feature_block_start.tolist(),[True,False,True,False,False,False,True,False])
        self.assertTrue(after.loc[after.feature_block_start,FEATURES].isna().all().all())
        self.assertTrue(after.speed_mps_valid.iloc[3])
        self.assertFalse(after.acceleration_mps2_valid.iloc[3])
        self.assertTrue(after.acceleration_mps2_valid.iloc[4])
        self.assertTrue(after.jerk_mps3_valid.iloc[5])
        self.assertEqual(after.break_reason.iloc[2],'speed_gt_limit')
        self.assertEqual(after.break_reason.iloc[6],'source_file_change')
        other=before.copy(); other['mode']='train'
        pd.testing.assert_series_equal(after.feature_block_id,repair_connections(other).feature_block_id)
        self.assertEqual(len(repair_connections(before.iloc[:0])),0)
