import unittest
import pandas as pd
from data_preprocessing.clean_trips import clean_user


class CleanTripsTests(unittest.TestCase):
    def test_threshold_files_duplicates_conflicts_and_invalid_points(self):
        d = pd.DataFrame({
            'user_id': ['001'] * 8,
            'source_file': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
            'point_id': list(range(8)),
            'timestamp': pd.to_datetime([0, 0, 0, 1200, 2401, 2402, 2402, 2403], unit='s'),
            'latitude': [1, 1, 2, 1, 1, 1, 1, 999],
            'longitude': [1] * 8,
            'altitude_ft': [-777] * 8,
            'mode': ['walk'] * 6 + ['bus', 'walk'],
            'label_status': ['matched'] * 8,
        })
        out, stats = clean_user(d)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.trip_id.tolist(), ['001_000001', '001_000001', '001_000002'])
        self.assertEqual(out.same_timestamp_input_count.tolist(), [3, 1, 1])
        self.assertEqual(stats['conflicting_timestamp_points_removed'], 2)
        self.assertEqual(stats['duplicate_observations_removed'], 1)
        self.assertEqual(stats['invalid_points_removed'], 1)
        self.assertTrue(out.altitude_ft.isna().all())
        empty, counts = clean_user(d.iloc[:0])
        self.assertEqual(counts['trips'], 0)
        self.assertEqual(len(empty), 0)
