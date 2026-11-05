import unittest
import numpy as np
import pandas as pd
from feature_enginneering.motion_features import build_features, EARTH_RADIUS_M, FEATURES


class MotionTests(unittest.TestCase):
    def test_irregular_sampling_derivatives_and_trip_boundaries(self):
        d = pd.DataFrame({'user_id': ['001']*5, 'trip_id': ['a']*4+['b'],
                          'timestamp': pd.to_datetime([0, 1, 3, 6, 2000], unit='s'),
                          'latitude': [0.]*5,
                          'longitude': np.rad2deg(np.array([0, 10, 50, 140, 200])/EARTH_RADIUS_M)})
        r = build_features(d)
        np.testing.assert_allclose(r.speed_mps.iloc[1:4], [10,20,30], atol=1e-7)
        np.testing.assert_allclose(r.acceleration_mps2.iloc[2:4], [5,10/3], atol=1e-7)
        self.assertAlmostEqual(r.jerk_mps3.iloc[3], -5/9, places=6)
        np.testing.assert_allclose(r.bearing_rad.iloc[1:4], np.pi/2)
        self.assertTrue(r.loc[4,FEATURES].isna().all())
        self.assertEqual(len(build_features(d.iloc[:0])), 0)

    def test_stationary_and_angle_wrap(self):
        # North-west followed by north-east: small turn across north, not 358 deg.
        d = pd.DataFrame({'user_id':['001']*4, 'trip_id':['a']*4,
                          'timestamp':pd.to_datetime([0,1,2,3],unit='s'),
                          'latitude':[0.,.001,.002,.002], 'longitude':[0.,-.00001,0.,0.]})
        r=build_features(d)
        self.assertLess(r.heading_change_rad.iloc[2], .03)
        self.assertEqual(r.speed_mps.iloc[3],0)
        self.assertFalse(r.bearing_rad_valid.iloc[3])
        self.assertFalse(r.heading_change_rad_valid.iloc[3])
