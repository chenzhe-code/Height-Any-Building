"""COCO bytes are a stable contract: compare with pycocotools' encoder."""
import sys
from pathlib import Path
import numpy as np
import unittest
import pycocotools.mask as mask_utils
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mls_mask_encoding import encode_binary_mask,encode_mask_results


def reference(mask):
    return mask_utils.encode(np.array(mask[:,:,None],order='F',dtype='uint8'))[0]


def check_binary_rle_bytes(shape):
    rng=np.random.RandomState(18)
    masks=[np.zeros(shape,bool),np.ones(shape,bool),rng.rand(*shape)<.35]
    for y,x in [(0,0),(-1,-1),(shape[0]//2,shape[1]//2)]:
        m=np.zeros(shape,bool);m[y,x]=True;masks.append(m)
    m=np.zeros(shape,bool);m[::2,::3]=True;masks.append(m)
    m=np.zeros(shape,bool);m[:,shape[1]//2]=True;masks.append(m)
    m=np.zeros(shape,bool);m[shape[0]//2,:]=True;masks.append(m)
    for m in masks:
        for view in [m,np.asfortranarray(m),m[::-1,::-1],m[::2,::2]]:
            encoded=encode_binary_mask(view)
            assert encoded==reference(view)
            np.testing.assert_array_equal(mask_utils.decode(encoded).astype(bool),view)


def test_dtype_fallback_and_mask_scoring_tuple():
    mask=np.zeros((15,18),dtype=np.uint8);mask[3:9,2:6]=1
    assert encode_binary_mask(mask)==reference(mask)
    scores=[[.9]]
    r=encode_mask_results(([[mask],[]],scores))
    assert r==([[reference(mask)],[]],scores)
    assert r[1] is scores
    assert encode_mask_results([[],[]])==[[],[]]

class EncodingTests(unittest.TestCase):
    def test_binary_arrays(self):
        for shape in [(1,1),(1,37),(41,1),(17,29),(1024,1024)]:
            with self.subTest(shape=shape):
                check_binary_rle_bytes(shape)

    def test_structure_and_dtype(self):
        test_dtype_fallback_and_mask_scoring_tuple()

if __name__ == '__main__':
    unittest.main()
