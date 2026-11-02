"""Structural counting tests only; no source/model refits or data generation."""
import unittest

import numpy as np

from latent_experiments.zhat_graph import structural_summary


class ZhatGraphTests(unittest.TestCase):
    def test_unordered_pairs_do_not_double_count_bidirection(self):
        p=np.zeros((4,4),dtype=int);p[1,2]=p[2,1]=1;p[0,1]=p[3,0]=1
        row=structural_summary(p)
        self.assertEqual(row['X_X_unordered_pairs'],1)
        self.assertEqual(row['X_X_directed_edges'],2)
        self.assertEqual(row['factor_connected_X_count'],2)
        self.assertEqual(row['factor_to_X_edges'],1)
        self.assertEqual(row['X_to_factor_edges'],1)

    def test_three_variable_reference_has_no_factor(self):
        p=np.zeros((3,3),dtype=int);p[0,2]=p[2,1]=1
        row=structural_summary(p,False)
        self.assertEqual(row['X_X_unordered_pairs'],2)
        self.assertIsNone(row['factor_connected_X_count'])

    def test_empty_graph_and_invalid_inputs(self):
        row=structural_summary(np.zeros((4,4)))
        self.assertEqual(row['factor_connection_coverage'],0)
        for p in (np.eye(4),np.full((4,4),.5),np.ones((3,4))):
            with self.assertRaises(ValueError):structural_summary(p)


if __name__=='__main__':unittest.main()
