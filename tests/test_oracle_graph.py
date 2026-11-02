"""Graph-metric unit checks only, no data generation or GraN-DAG training."""
import unittest

import numpy as np

from latent_experiments.oracle import graph_scores


class OracleGraphTests(unittest.TestCase):
    def setUp(self):
        self.truth=np.zeros((4,4),dtype=int);self.truth[0,1:]=1

    def test_perfect_and_empty(self):
        a=graph_scores(self.truth,self.truth)
        self.assertEqual((a['shd'],a['precision'],a['recall'],a['f1']),(0,1,1,1))
        b=graph_scores(np.zeros((4,4)),self.truth)
        self.assertEqual((b['shd'],b['missing_true_edges'],b['f1']),(3,3,0))

    def test_reversal_costs_one_but_precision_counts_directed_edges(self):
        pred=self.truth.copy();pred[0,1]=0;pred[1,0]=1
        a=graph_scores(pred,self.truth)
        self.assertEqual((a['shd'],a['reversed_edges'],a['missing_true_edges'],a['extra_directed_edges']),(1,1,1,1))
        self.assertAlmostEqual(a['precision'],2/3);self.assertAlmostEqual(a['recall'],2/3)

    def test_bidirectional_prediction_is_extra_deletion_not_reversal(self):
        pred=self.truth.copy();pred[1,0]=1
        a=graph_scores(pred,self.truth)
        self.assertEqual((a['shd'],a['reversed_edges'],a['correct_edges'],a['extra_without_reversal']),(1,0,3,1))

    def test_extra_observed_edge_and_invalid_matrices(self):
        pred=self.truth.copy();pred[1,2]=1
        a=graph_scores(pred,self.truth)
        self.assertEqual(a['shd'],1);self.assertEqual(a['extra_directed_edges'],1)
        for invalid in (np.eye(4),np.ones((3,3)),np.full((4,4),.5)):
            with self.assertRaises(ValueError):graph_scores(invalid,self.truth)


if __name__=='__main__':unittest.main()
