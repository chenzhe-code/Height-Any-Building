"""Preserve ordered float64 accumulation, coverage and exceptional values."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.io_utils import _mean_mosaic


def reference(patches,positions,shape):
    out=np.zeros(shape);count=np.zeros_like(out)
    for patch,(x,y) in zip(patches,positions):
        h,w=patch.shape
        out[y:y+h,x:x+w]+=patch
        count[y:y+h,x:x+w]+=1
    return np.divide(out,count,out=np.zeros_like(out),where=count>0)


class MeanTests(unittest.TestCase):
    def test_order_dtype_irregular_overlaps_and_holes(self):
        rng=np.random.RandomState(1234)
        positions=[(3,4),(0,0),(5,1),(1,2),(3,4),(18,14)]
        patches=[rng.normal(size=(7,8)).astype(dtype) for dtype in
                 [np.float32,np.float64,np.int16,np.float32,np.float64,np.float32]]
        # Cancellation makes a reassociated or float32 reduction detectable.
        patches[0][:]=1e12;patches[4][:]=-1e12
        original=[p.copy() for p in patches];expected=reference(patches,positions,(23,30))
        for block in [1,3,7,256,512]:
            actual=_mean_mosaic(patches,positions,(23,30),block_size=block)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(actual.dtype,np.float64)
        for x,y in zip(original,patches):np.testing.assert_array_equal(x,y)

    def test_nonfinite_and_zero_values(self):
        patches=[np.array([[0.,-0.,np.nan],[np.inf,-np.inf,1.]]),
                 np.array([[-0.,0.,5.],[-np.inf,np.inf,2.]])]
        with np.errstate(invalid='ignore'):
            actual=_mean_mosaic(patches,[(0,0)]*2,(3,4),block_size=1)
            expected=reference(patches,[(0,0)]*2,(3,4))
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))

    def test_invalid_geometry(self):
        for args in [([np.ones((2,3))],[],(3,4)),
                     ([np.ones((2,3))],[(-1,0)],(3,4)),
                     ([np.ones((2,3))],[(2,2)],(3,4)),
                     ([np.ones((2,3,1))],[(0,0)],(3,4))]:
            with self.assertRaises(ValueError):_mean_mosaic(*args)
        with self.assertRaises(ValueError):_mean_mosaic([],[],(3,4),block_size=0)
        np.testing.assert_array_equal(_mean_mosaic([],[],(3,4)),np.zeros((3,4)))

if __name__=='__main__':unittest.main()
