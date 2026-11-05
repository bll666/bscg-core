"""Data-free unit checks on mathematical operations and architecture contracts."""
import unittest
import numpy as np
import torch
from bscg.model import BSCG, parameter_count, noisy_view, reorder_blocks
from bscg.preprocess import Standardizer, interpolate_short_gaps
from bscg.metrics import per_cow, interval
from bscg.splits import make_splits, check_splits


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_parameter_totals(self):
        self.assertEqual(parameter_count(BSCG(10,10,30)), 29825)
        self.assertEqual(parameter_count(BSCG(1,30,2)), 29889)
        self.assertEqual(parameter_count(BSCG(10,10,30,local_layers=0)), 21281)
        self.assertEqual(parameter_count(BSCG(10,10,30,global_layers=0)), 12737)

    def test_forward_and_gradient(self):
        m = BSCG(1,30,2)
        x, mask, static = torch.ones(2,12,1), torch.ones(2,12,1), torch.zeros(2,30)
        p,h = m(x,mask,static)
        loss = p.square().mean()+.02*m.auxiliary(h,x,mask,[0]).mean()
        loss.backward()
        self.assertEqual(tuple(p.shape),(2,))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None))

    def test_noise_respects_masks_and_binary_channel(self):
        x = torch.ones(1,12,2); mask = torch.ones_like(x); mask[:,0,0]=0
        p = noisy_view(x,mask,[0])
        self.assertTrue(torch.equal(x[:,:,1],p[:,:,1]))
        self.assertEqual(float(p[0,0,0]),1.)

    def test_block_alignment(self):
        x = torch.arange(12.).reshape(1,12,1); mask = x+100
        order = torch.tensor([[5,4,3,2,1,0]])
        a,b = reorder_blocks(x,mask,2,order)
        self.assertTrue(torch.equal(b-a,torch.full_like(x,100)))
        self.assertEqual(a.flatten().tolist(),[10,11,8,9,6,7,4,5,2,3,0,1])

    def test_invalid_segment_length(self):
        with self.assertRaises(ValueError):
            BSCG(1,30,5)(torch.ones(1,12,1),torch.ones(1,12,1),torch.ones(1,30))


class TransformTests(unittest.TestCase):
    def test_fit_uses_only_passed_training_values(self):
        s = Standardizer().fit(np.array([[0.,2.],[2.,4.]]))
        self.assertTrue(np.array_equal(s.mean,[1.,3.]))
        result = s.transform(np.array([[1000.,np.nan]]))
        self.assertTrue(np.array_equal(result,[[5.,0.]]))
        restored = Standardizer.restore(s.state())
        self.assertTrue(np.array_equal(restored.transform([[1.,3.]]),[[0.,0.]]))

    def test_complete_short_gap_only(self):
        a = np.arange(10.).reshape(1,10,1); a[0,2:4,0]=np.nan; a[0,5:8,0]=np.nan
        b = interpolate_short_gaps(a,[0],2)
        self.assertTrue(np.array_equal(b[0,2:4,0],[2,3]))
        self.assertTrue(np.isnan(b[0,5:8,0]).all())

    def test_no_boundary_extrapolation(self):
        a = np.array([[[np.nan],[1.],[2.],[np.nan]]])
        b = interpolate_short_gaps(a,[0],2)
        self.assertTrue(np.isnan(b[0,0,0]) and np.isnan(b[0,-1,0]))


class MetricTests(unittest.TestCase):
    def test_macro_is_equal_cow_not_equal_record(self):
        _,a,b=per_cow([1.,1.,5.],[0.,0.,0.],["a","a","b"])
        self.assertEqual(a.mean(),3.)
        self.assertEqual(b.mean(),3.)

    def test_score_mean_prediction_separately(self):
        seeds=np.array([[1.],[3.]]); truth=np.array([2.])
        self.assertEqual(float(np.abs(seeds.mean(0)-truth).mean()),0.)
        self.assertEqual(float(np.abs(seeds-truth).mean()),1.)

    def test_bootstrap_reproducible(self):
        self.assertEqual(interval([1.,2.,4.],200,17),interval([1.,2.,4.],200,17))


class SplitTests(unittest.TestCase):
    def test_nested_cow_and_time_boundaries(self):
        cows=np.repeat([f"C{i:02d}" for i in range(1,11)],12)
        dates=np.tile(np.arange(np.datetime64("2023-07-23"),np.datetime64("2023-08-04")).astype(str),10)
        d=dict(cows=cows,input_dates=dates,
               target_dates=(dates.astype("datetime64[D]")+np.timedelta64(1,"D")).astype(str))
        cfg=dict(dataset="mmcows",input_start="2023-07-23",input_end="2023-08-03",
                 fit_end="2023-07-28",test_start="2023-07-30")
        loco=make_splits(d,cfg,"mm_loco");check_splits(d,loco)
        self.assertEqual(len(loco),10)
        self.assertEqual(len(loco[0]["train"]),108)
        self.assertEqual(len(loco[0]["inner"][0]["validation"]),36)
        t=make_splits(d,cfg,"mm_time");b=make_splits(d,cfg,"mm_cow_time")
        check_splits(d,t);check_splits(d,b)
        self.assertEqual(len(t[0]["tests"]["MM_TIME"]),50)
        self.assertTrue(np.array_equal(np.sort(t[0]["tests"]["MM_TIME"]),
            np.sort(np.concatenate([f["tests"]["MM_BOTH"] for f in b]))))
        self.assertEqual(len(b[0]["train"]),54)


if __name__ == "__main__": unittest.main()
